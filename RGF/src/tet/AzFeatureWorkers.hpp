/* Persistent workers for independent feature searches. MIT license. */
#ifndef AZ_FEATURE_WORKERS_HPP
#define AZ_FEATURE_WORKERS_HPP

#include "AzUtil.hpp"
#include <algorithm>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

class AzFeatureWorkers {
  std::vector<std::thread> workers;
  std::vector<std::exception_ptr> errors;
  std::mutex mutex;
  std::condition_variable ready, finished;
  std::function<void(int)> job;
  unsigned long generation;
  int pending;
  bool stopping;

  void perform(int worker) {
    try { job(worker); }
    catch (...) { errors[worker] = std::current_exception(); }
  }

  void work(int worker) {
    unsigned long seen = 0;
    std::unique_lock<std::mutex> lock(mutex);
    for (;;) {
      ready.wait(lock, [&] { return stopping || generation != seen; });
      if (stopping) return;
      seen = generation;
      lock.unlock();
      perform(worker);
      lock.lock();
      if (--pending == 0) finished.notify_one();
    }
  }

  void stop() {
    { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
    ready.notify_all();
    for (size_t ix = 0; ix < workers.size(); ++ix) workers[ix].join();
  }

public:
  explicit AzFeatureWorkers(int count) : errors(count), generation(0),
                                         pending(0), stopping(false) {
    try {
      workers.reserve(count-1);
      for (int ix = 1; ix < count; ++ix)
        workers.push_back(std::thread(&AzFeatureWorkers::work, this, ix));
    }
    catch (...) { stop(); throw; }
  }
  ~AzFeatureWorkers() { stop(); }
  int size() const { return static_cast<int>(errors.size()); }

  void run(const std::function<void(int)> &function) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      job = function;
      std::fill(errors.begin(), errors.end(), std::exception_ptr());
      pending = static_cast<int>(workers.size());
      ++generation;
    }
    ready.notify_all();
    perform(0);
    {
      std::unique_lock<std::mutex> lock(mutex);
      finished.wait(lock, [&] { return pending == 0; });
      job = std::function<void(int)>();
    }
    // Join the whole search before propagating an error to the training thread.
    std::exception_ptr first;
    for (size_t ix = 0; ix < errors.size(); ++ix) {
      if (!errors[ix]) continue;
      if (!first) first = errors[ix];
      else {
        try { std::rethrow_exception(errors[ix]); }
        catch (AzException *error) { delete error; }
        catch (...) {}
      }
    }
    if (first) std::rethrow_exception(first);
  }
};
#endif
