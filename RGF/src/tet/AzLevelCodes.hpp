/* One-byte level codes for exact histogram split search. MIT license. */
#ifndef AZ_LEVEL_CODES_HPP
#define AZ_LEVEL_CODES_HPP

#include "AzUtil.hpp"
#include "AzSmat.hpp"
#include "AzFeatureWorkers.hpp"
#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#define Az_max_level_num 256

/*
 * When every feature has at most 256 distinct values, each training value is
 * kept as one byte: its rank among that feature's distinct values.  Codes are
 * row-major (data point x feature), so one pass over a node's rows fills the
 * per-level sums of all features.
 *
 * Exactness: the sorted lists order rows by (value, row).  Summing each level
 * over the node's rows in ascending row order adds the same numbers in the
 * same order as the sorted path, and partition() hands the children their
 * rows in (value, row) order as the sorted lists do, so models are identical.
 */
class AzLevelCodes {
protected:
  int data_num, feat_num;
  std::unique_ptr<unsigned char[]> codes; /* data_num x feat_num */
  std::vector<double> values;             /* ascending distinct values per feature */
  std::vector<int> offsets;               /* feat_num+1 offsets into values */

public:
  AzLevelCodes() : data_num(0), feat_num(0) {}

  bool isActive() const { return codes.get() != NULL; }
  void reset() {
    data_num = feat_num = 0;
    codes.reset();
    std::vector<double>().swap(values);
    std::vector<int>().swap(offsets);
  }

  int featNum() const { return feat_num; }
  int levelNum(int fx) const { return offsets[fx+1] - offsets[fx]; }
  const double *levelValues(int fx) const { return values.data() + offsets[fx]; }
  const int *levelOffsets() const { return offsets.data(); }
  const double *allLevelValues() const { return values.data(); }
  const unsigned char *rowCodes(int dx) const {
    return codes.get() + (size_t)dx*feat_num;
  }
  double value(int dx, int fx) const {
    return values[offsets[fx] + codes[(size_t)dx*feat_num + fx]];
  }
  int maxLevelNum() const {
    int level_max = 0;
    for (int fx = 0; fx < feat_num; ++fx) level_max = MAX(level_max, levelNum(fx));
    return level_max;
  }
  double codeBytes() const { return (double)data_num*feat_num; }

  /*
   * Build codes from data (rows: features, columns: data points).  Returns
   * false and keeps nothing when a value is NaN, a feature has more than 256
   * distinct values, or a column is not in strict row order.
   */
  bool build(const AzSmat *m_data, int thread_num) {
    reset();
    const int f_num = m_data->rowNum();
    const int d_num = m_data->colNum();
    if (f_num <= 0 || d_num <= 0) return false;

    const int worker_count = MAX(1, MIN(thread_num, d_num));
    std::unique_ptr<AzFeatureWorkers> workers;
    if (worker_count > 1) workers.reset(new AzFeatureWorkers(worker_count));
    const auto run = [&](const std::function<void(int)> &job) {
      if (workers) workers->run(job);
      else job(0);
    };

    /*---  distinct nonzero values and nonzero counts per feature, per worker  ---*/
    std::vector<std::vector<std::vector<double> > > found(worker_count);
    std::vector<std::vector<int> > nonzero(worker_count);
    std::vector<char> usable(worker_count, 1);
    run([&](int worker) {
      std::vector<std::vector<double> > &lists = found[worker];
      std::vector<int> &counts = nonzero[worker];
      lists.resize(f_num);
      counts.assign(f_num, 0);
      const int begin = (int)((long long)d_num*worker/worker_count);
      const int end = (int)((long long)d_num*(worker+1)/worker_count);
      for (int dx = begin; dx < end; ++dx) {
        int num = 0;
        const AZI_VECT_ELM *elm = m_data->col(dx)->storedElements(&num);
        int prev_fx = -1;
        for (int ex = 0; ex < num; ++ex) {
          const int fx = elm[ex].no;
          const double val = elm[ex].val;
          if (fx <= prev_fx || fx >= f_num || val != val) {
            usable[worker] = 0;
            return;
          }
          prev_fx = fx;
          if (val == 0) continue; /* read as 0, as AzSvect::next does */
          ++counts[fx];
          std::vector<double> &list = lists[fx];
          std::vector<double>::iterator it = std::lower_bound(list.begin(), list.end(), val);
          if (it != list.end() && *it == val) continue;
          if ((int)list.size() >= Az_max_level_num) {
            usable[worker] = 0;
            return;
          }
          list.insert(it, val);
        }
      }
    });
    for (int wx = 0; wx < worker_count; ++wx) {
      if (!usable[wx]) return false;
    }

    /*---  merge into one ascending table per feature  ---*/
    std::vector<unsigned char> zero_code(f_num, 0);
    std::vector<double> merged;
    offsets.assign(f_num+1, 0);
    for (int fx = 0; fx < f_num; ++fx) {
      merged.clear();
      long long count = 0;
      for (int wx = 0; wx < worker_count; ++wx) {
        merged.insert(merged.end(), found[wx][fx].begin(), found[wx][fx].end());
        count += nonzero[wx][fx];
        std::vector<double>().swap(found[wx][fx]);
      }
      const bool has_zero = (count < d_num);
      if (has_zero) merged.push_back(0);
      std::sort(merged.begin(), merged.end());
      merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
      if ((int)merged.size() > Az_max_level_num) {
        reset();
        return false;
      }
      if (has_zero) {
        zero_code[fx] = (unsigned char)(std::lower_bound(merged.begin(), merged.end(), 0.0)
                                        - merged.begin());
      }
      offsets[fx+1] = offsets[fx] + (int)merged.size();
      values.insert(values.end(), merged.begin(), merged.end());
    }

    /*---  codes  ---*/
    data_num = d_num;
    feat_num = f_num;
    codes.reset(new unsigned char[(size_t)d_num*f_num]);
    run([&](int worker) {
      const int begin = (int)((long long)d_num*worker/worker_count);
      const int end = (int)((long long)d_num*(worker+1)/worker_count);
      const double *table = values.data();
      for (int dx = begin; dx < end; ++dx) {
        unsigned char *row = codes.get() + (size_t)dx*f_num;
        memcpy(row, zero_code.data(), f_num);
        int num = 0;
        const AZI_VECT_ELM *elm = m_data->col(dx)->storedElements(&num);
        for (int ex = 0; ex < num; ++ex) {
          const double val = elm[ex].val;
          if (val == 0) continue;
          const int fx = elm[ex].no;
          const double *first = table + offsets[fx];
          const double *last = table + offsets[fx+1];
          row[fx] = (unsigned char)(std::lower_bound(first, last, val) - first);
        }
      }
    });
    return true;
  }

  /*
   * A node's rows in ascending order.  Returns dxs itself when it is already
   * ascending; otherwise merges its ascending runs (one per level of the
   * parent's split feature) into *buffer.
   */
  static const int *ascendingRows(const int *dxs, int dxs_num, std::vector<int> *buffer) {
    std::vector<int> run_begin(1, 0);
    for (int ix = 1; ix < dxs_num; ++ix) {
      if (dxs[ix] < dxs[ix-1]) run_begin.push_back(ix);
    }
    if (run_begin.size() == 1) return dxs;
    run_begin.push_back(dxs_num);

    buffer->assign(dxs, dxs + dxs_num);
    std::vector<int> other(dxs_num);
    std::vector<int> *src = buffer, *dst = &other;
    std::vector<int> next_begin;
    while (run_begin.size() > 2) {
      next_begin.clear();
      size_t rx = 0;
      for ( ; rx + 2 < run_begin.size(); rx += 2) {
        std::merge(src->begin() + run_begin[rx], src->begin() + run_begin[rx+1],
                   src->begin() + run_begin[rx+1], src->begin() + run_begin[rx+2],
                   dst->begin() + run_begin[rx]);
        next_begin.push_back(run_begin[rx]);
      }
      if (rx + 1 < run_begin.size()) { /* odd run out */
        std::copy(src->begin() + run_begin[rx], src->begin() + run_begin[rx+1],
                  dst->begin() + run_begin[rx]);
        next_begin.push_back(run_begin[rx]);
      }
      next_begin.push_back(dxs_num);
      run_begin.swap(next_begin);
      std::swap(src, dst);
    }
    if (src != buffer) buffer->swap(other);
    return buffer->data();
  }

  /*
   * Split a node's rows at border_val.  Each side lists its rows by (value,
   * row), as AzSortedFeat_Dense::getIndexes does with its "> border" test.
   */
  void partition(const int *dxs, int dxs_num, int fx, double border_val,
                 AzIntArr *ia_le, AzIntArr *ia_gt) const {
    std::vector<int> buffer;
    const int *rows = ascendingRows(dxs, dxs_num, &buffer);
    const int level_num = levelNum(fx);
    const double *level_value = levelValues(fx);
    const unsigned char *code = codes.get() + fx;

    std::vector<int> position(level_num+1, 0);
    for (int ix = 0; ix < dxs_num; ++ix) ++position[code[(size_t)rows[ix]*feat_num] + 1];
    for (int lx = 0; lx < level_num; ++lx) position[lx+1] += position[lx];

    int le_level = 0;
    while (le_level < level_num && !(level_value[le_level] > border_val)) ++le_level;
    const int le_num = position[le_level];

    std::vector<int> ordered(dxs_num);
    for (int ix = 0; ix < dxs_num; ++ix) {
      const int dx = rows[ix];
      ordered[position[code[(size_t)dx*feat_num]]++] = dx;
    }
    ia_le->reset(ordered.data(), le_num);
    ia_gt->reset(ordered.data() + le_num, dxs_num - le_num);
  }
};

#endif
