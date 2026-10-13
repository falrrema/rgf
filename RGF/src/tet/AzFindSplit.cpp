/* * * * *
 *  AzFindSplit.cpp 
 *  Copyright (C) 2011, 2012 Rie Johnson, 2018 RGF-team
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the COPYING file for details.
 * * * * */

#include "AzFindSplit.hpp"

/*--------------------------------------------------------*/
void AzFindSplit::_begin(const AzTrTree_ReadOnly *inp_tree, 
                         const AzDataForTrTree *inp_data, 
                         const AzTrTtarget *inp_target, 
                         int inp_min_size)
{
  tree = inp_tree; 
  target = inp_target; 
  min_size = inp_min_size; 
  data = inp_data; 
}

/*--------------------------------------------------------*/
void AzFindSplit::_findBestSplit(int nx, 
                                 /*---  output  ---*/
                                 AzTrTsplit *best_split)
{
  const char *eyec = "AzFindSplit::_findBestSplit"; 

  if (tree == NULL || target == NULL || data == NULL) {
    throw new AzException(eyec, "information is not set");
  }

  if (replay_split_cache) {
    replayBestSplit(best_split);
    return;
  }

  const int *dxs = tree->node(nx)->data_indexes();
  const int dxs_num = tree->node(nx)->dxs_num; 

  const int worker_count = std::min(split_threads, data->featNum());
  const bool parallel = worker_count > 1 &&
                        static_cast<long long>(dxs_num)*data->featNum() >= 32768;
  if (parallel && (!feature_workers || feature_workers->size() != worker_count))
    feature_workers.reset(new AzFeatureWorkers(worker_count));
  const AzSortedFeatArr *sorted_arr = tree->sorted_array(
      nx, data, parallel ? feature_workers.get() : NULL);
  if (sorted_arr == NULL) {
    throw new AzException(eyec, "No sorted array?!"); 
  }

  Az_forFindSplit total; 
  total.wy_sum = target->getTarDwSum(dxs, dxs_num);
  total.w_sum = target->getDwSum(dxs, dxs_num); 

  /*---  go through features to find the best split  ---*/
  int feat_num = data->featNum(); 
  const int *fxs = NULL; 
  if (ia_fx != NULL) {
    fxs = ia_fx->point(&feat_num); 
  }
  // Contiguous feature ranges preserve the original first-wins tie rule.
  // Small nodes stay serial to avoid dispatch overhead; no arithmetic changes.
  std::unique_ptr<AzTrTsplit[]> candidates;
  if (parallel) candidates.reset(new AzTrTsplit[worker_count]);
  // Each worker records its own feature range; concatenating the records in
  // worker order reproduces the serial evaluation order exactly.
  const int record_limit = splitCacheLimit();
  std::vector<std::vector<AzSplitCandidate> > records(split_cache != NULL ? (parallel ? worker_count : 1) : 0);
  const auto search = [&](int worker) {
    AzTrTsplit *candidate = parallel ? &candidates[worker] : best_split;
    if (parallel) candidate->copy(best_split);
    std::vector<AzSplitCandidate> *record = records.empty() ? NULL : &records[worker];
    const int begin = parallel ? static_cast<long long>(feat_num)*worker/worker_count : 0;
    const int end = parallel ? static_cast<long long>(feat_num)*(worker+1)/worker_count : feat_num;
    for (int ix = begin; ix < end; ++ix) {
      int fx = ix;
      if (fxs != NULL) fx = fxs[ix];

      AzSortedFeatWork tmp;
      const AzSortedFeat *sorted = sorted_arr->sorted(fx);
      if (sorted == NULL) { /* This happens only with Thrift or warm-start */
        const AzSortedFeat *my_sorted = sorted_arr->sorted(data->sorted_array(), fx, &tmp);
        if (my_sorted->dataNum() != dxs_num) {
          throw new AzException(eyec, "conflict in #data");
        }
        loop(candidate, fx, my_sorted, dxs_num, &total, record, record_limit);
      }
      else {
        loop(candidate, fx, sorted, dxs_num, &total, record, record_limit);
      }
    }
  };
  if (parallel) {
    if (!feature_workers || feature_workers->size() != worker_count)
      feature_workers.reset(new AzFeatureWorkers(worker_count));
    feature_workers->run(search);
    for (int worker = 0; worker < worker_count; ++worker)
      if (candidates[worker].gain > best_split->gain)
        best_split->copy(&candidates[worker]);
  }
  else search(0);

  if (split_cache != NULL) {
    size_t count = 0;
    bool complete = true;
    for (size_t wx = 0; wx < records.size(); ++wx) {
      count += records[wx].size();
      if (records[wx].size() > (size_t)record_limit) complete = false;
    }
    split_cache->clear();
    if (complete && count <= (size_t)record_limit) {
      split_cache->candidates.reserve(count);
      for (size_t wx = 0; wx < records.size(); ++wx) {
        split_cache->candidates.insert(split_cache->candidates.end(),
                                       records[wx].begin(), records[wx].end());
      }
      split_cache->valid = true;
    }
  }

  if (best_split->fx >= 0) {
    if (!dmp_out.isNull()) {
      data->featInfo()->desc(best_split->fx, &best_split->str_desc); 
    }
  }
}

/*--------------------------------------------------------*/
double AzFindSplit::evalSplit(const Az_forFindSplit i[2],
                              double bestP[2]) const
{
  double gain = getBestGain(i[0].w_sum, i[0].wy_sum, &bestP[0]);
  gain += getBestGain(i[1].w_sum, i[1].wy_sum, &bestP[1]);
  return gain;
}

/*--------------------------------------------------------*/
void AzFindSplit::loop(AzTrTsplit *best_split,
                       int fx, /* feature# */
                       const AzSortedFeat *sorted,
                       int total_size,
                       const Az_forFindSplit *total,
                       std::vector<AzSplitCandidate> *record, /* may be NULL */
                       int record_limit)
{
  /*---  first everyone is in GT(LE)  ---*/
  /*---  move the smallest(largest) ones from GT(LE) to LE(GT)  ---*/

  int dest_size = 0; 
  Az_forFindSplit i[2];
  Az_forFindSplit *src = &i[1];
  Az_forFindSplit *dest = &i[0];

  int le_idx, gt_idx;
  if (sorted->isForward()) {
    le_idx = 0; 
    gt_idx = 1; 
  }
  else {
    le_idx = 1; 
    gt_idx = 0; 
  }

  AzCursor cursor; 
  sorted->rewind(cursor); 

  for ( ; ; ) {
    double value; 
    int index_num; 
    const int *index = NULL; 
    index = sorted->next(cursor, &value, &index_num); 
    if (index == NULL) break; 
    dest_size += index_num;  
    if (dest_size >= total_size) {
      break; /* don't allow all vs nothing */
    }

    const double *tarDw = target->tarDw_arr(); 
    const double *dw = target->dw_arr(); 
    double wy_sum_move = 0, w_sum_move = 0; 
    for (int ix = 0; ix < index_num; ++ix) {
      int dx = index[ix]; 
      wy_sum_move += tarDw[dx]; 
      w_sum_move += dw[dx]; 
    }
    dest->wy_sum += wy_sum_move; 
    dest->w_sum += w_sum_move; 

    if (min_size > 0) {
      if (dest_size < min_size) {
        continue; 
      }
      if (total_size - dest_size < min_size) {
        break; 
      }
    }

    src->wy_sum = total->wy_sum - dest->wy_sum;
    src->w_sum  = total->w_sum  - dest->w_sum;

    /* one record past the limit marks the cache as incomplete */
    if (record != NULL && (int)record->size() <= record_limit) {
      AzSplitCandidate c;
      c.i[0] = i[0]; c.i[1] = i[1];
      c.value = value; c.fx = fx; c.forward = (le_idx == 0);
      record->push_back(c);
    }

    double bestP[2] = {0, 0};
    const double gain = evalSplit(i, bestP);
    if (gain > best_split->gain) {
      best_split->reset_values(fx, value, gain,
                               bestP[le_idx], bestP[gt_idx]);
    }
  }
}

/*--------------------------------------------------------*/
/* Same evalSplit calls, in the same order, on the same recorded sums. */
void AzFindSplit::replayBestSplit(AzTrTsplit *best_split) const
{
  const std::vector<AzSplitCandidate> &list = split_cache->candidates;
  for (size_t cx = 0; cx < list.size(); ++cx) {
    const AzSplitCandidate &c = list[cx];
    double bestP[2] = {0, 0};
    const double gain = evalSplit(c.i, bestP);
    if (gain > best_split->gain) {
      best_split->reset_values(c.fx, c.value, gain,
                               bestP[c.forward ? 0 : 1], bestP[c.forward ? 1 : 0]);
    }
  }
  if (best_split->fx >= 0) {
    if (!dmp_out.isNull()) {
      data->featInfo()->desc(best_split->fx, &best_split->str_desc);
    }
  }
}

/*--------------------------------------------------------*/
void AzFindSplit::_pickFeats(int pick_num, int f_num)
{
  if (pick_num < 1 || pick_num > f_num) {
    throw new AzException("AzFindSplit::pickFeats", "out of range");
  }
  ia_feats.reset();
  if (pick_num == f_num) {
    ia_fx = NULL;
    return;
  }

  AzIntArr ia_onOff;
  ia_onOff.reset(f_num, 0);
  int *onOff = ia_onOff.point_u();
  for ( ; ; ) {
    if (ia_feats.size() >= pick_num) break;
    int fx = rand() % f_num;
    if (onOff[fx] == 0) {
      onOff[fx] = 1;
      ia_feats.put(fx);
    }
  }
  ia_fx = &ia_feats;
}
