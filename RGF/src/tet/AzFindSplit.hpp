/* * * * *
 *  AzFindSplit.hpp 
 *  Copyright (C) 2011, 2012 Rie Johnson, 2018 RGF-team
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the COPYING file for details.
 * * * * */

#ifndef _AZ_FIND_SPLIT_HPP_
#define _AZ_FIND_SPLIT_HPP_

#include "AzUtil.hpp"
#include "AzDataForTrTree.hpp"
#include "AzTrTtarget.hpp"
#include "AzTrTsplit.hpp"
#include "AzTrTree.hpp"
#include "AzFeatureWorkers.hpp"
#include "AzSplitCache.hpp"
#include "AzLevelCodes.hpp"
#include <memory>
#include <vector>

/* Target sums of one level (distinct value) of one feature within a node. */
class AzLevelBin {
public:
  double wy_sum, w_sum;
  int count;
  AzLevelBin() : wy_sum(0), w_sum(0), count(0) {}
};

/* One worker's level sums for its features, reused across nodes. */
class AzLevelWork {
public:
  std::vector<AzLevelBin> bins;
  std::vector<int> first_bin; /* per feature position */
};

//! Abstract class: provides building blocks for node split search. 
/*------------------------------------------*/
class AzFindSplit 
{
protected:
  const AzTrTtarget *target; 
  const AzDataForTrTree *data; 
  const AzTrTree_ReadOnly *tree; 
  int min_size;
  int split_threads;
  std::unique_ptr<AzFeatureWorkers> feature_workers;
  bool split_cache_enabled;
  AzSplitCache *split_cache; /* record into it, or replay it when replay_split_cache */
  bool replay_split_cache;

  AzIntArr ia_feats;
  const AzIntArr *ia_fx;

  std::vector<AzLevelWork> level_work; /* per worker */
  std::vector<int> ascending_rows;

public:
  AzFindSplit() : target(NULL), data(NULL), tree(NULL), ia_fx(NULL),
                  min_size(-1), split_threads(1), split_cache_enabled(true),
                  split_cache(NULL), replay_split_cache(false) {}
  virtual ~AzFindSplit() {}
  void reset() {
    target = NULL;
    data = NULL; 
    tree = NULL;  
    min_size = -1; 
  }

  void _begin(const AzTrTree_ReadOnly *inp_tree, 
              const AzDataForTrTree *inp_data, 
              const AzTrTtarget *inp_target, 
              int inp_min_size); 
  void _end() {
    reset(); 
  }

  //----------------------------------------------------------------
  //  void findBestSplit(const AzTrTtarget *tar, 
  //                 const AzIntArr *ia_dx,              
  //                 ... parameters ...
  //                 AzTrTsplit *best_split); /* output */
  //----------------------------------------------------------------

  virtual void _pickFeats(int pick_num, int f_num);

  /*
   * Use a per-leaf candidate cache for the next search.  A random feature
   * subset changes the candidates between searches, so it disables the cache.
   */
  void useSplitCacheFor(AzSplitCache *cache, bool replay) {
    if (!split_cache_enabled || ia_fx != NULL) cache = NULL;
    split_cache = cache;
    replay_split_cache = (cache != NULL && replay && cache->valid);
  }

protected: 
  /*----------------------------------------------------------------*/
  virtual double getBestGain(double w_sum, 
                             double wy_sum, 
                             double *out_best_p) /* must not be null */
                             const = 0; 
  virtual double evalSplit(const Az_forFindSplit i[2], 
                           double bestP[2]) /* output */ const;
  /*----------------------------------------------------------------*/

  void _findBestSplit(int nx, 
                      /*---  output  ---*/
                      AzTrTsplit *best_split); 
  void loop(AzTrTsplit *best_split,
            int fx, /* feature# */
            const AzSortedFeat *sorted,
            int dxs_num,
            const Az_forFindSplit *total,
            std::vector<AzSplitCandidate> *record, /* may be NULL */
            int record_limit);
  void replayBestSplit(AzTrTsplit *best_split) const;
  void levelSearch(AzTrTsplit *best_split,
                   const AzLevelCodes *level_codes,
                   const int *fxs, /* may be NULL */
                   int begin, int end, /* positions in fxs or feature#s */
                   const int *rows, /* ascending */
                   int row_num,
                   const Az_forFindSplit *total,
                   std::vector<AzSplitCandidate> *record, /* may be NULL */
                   int record_limit,
                   AzLevelWork *work);
  void levelLoop(AzTrTsplit *best_split,
                 int fx,
                 const AzLevelBin *bins,
                 const double *level_value,
                 int level_num,
                 int total_size,
                 const Az_forFindSplit *total,
                 std::vector<AzSplitCandidate> *record, /* may be NULL */
                 int record_limit);
  int splitCacheLimit() const {
    return MAX(4096, 16*data->featNum());
  }
};

#endif 
