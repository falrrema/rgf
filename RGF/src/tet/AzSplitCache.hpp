/* Recorded split candidates for exact re-evaluation of unchanged leaves. MIT license. */
#ifndef AZ_SPLIT_CACHE_HPP
#define AZ_SPLIT_CACHE_HPP

#include <vector>

class Az_forFindSplit {
public:
  double wy_sum, w_sum;
  Az_forFindSplit() : wy_sum(0), w_sum(0) {}
  void reset() {
    wy_sum = w_sum = 0;
  }
};

/*
 * One evaluated threshold: the exact sums passed to evalSplit and what is needed
 * to rebuild the split.  The sums depend only on the rows of the leaf and their
 * gradients/weights, so they stay valid until those rows' targets change.
 */
class AzSplitCandidate {
public:
  Az_forFindSplit i[2];
  double value;
  int fx;
  bool forward;
};

class AzSplitCache {
public:
  bool valid;
  std::vector<AzSplitCandidate> candidates;
  AzSplitCache() : valid(false) {}
  void clear() {
    valid = false;
    std::vector<AzSplitCandidate>().swap(candidates);
  }
};

#endif
