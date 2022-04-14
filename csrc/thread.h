#pragma once

#include <condition_variable>
#include <future>
#include <queue>
#include <thread>
#include <iostream>

// A simple C++11 Thread Pool implementation with `num_workers=1`.
// See: https://github.com/progschj/ThreadPool
class Thread {
public:
  Thread();
  ~Thread();
  template <class F> void run(F &&f);
  void synchronize();

private:
  bool stop;
  std::mutex mutex;
  std::thread worker;
  std ::condition_variable condition;
  std::queue<std::future<void>> results;
  std::queue<std::function<void()>> tasks;
};

inline Thread::Thread() : stop(false) {
  worker = std::thread([this] {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(this->mutex);
        this->condition.wait(
            lock, [this] { return this->stop || !this->tasks.empty(); });
        if (this->stop && this->tasks.empty())
          return;
        task = std::move(this->tasks.front());
        this->tasks.pop();
      }
      task();
    }
  });
}

inline Thread::~Thread() {
  {
    std::unique_lock<std::mutex> lock(mutex);
    stop = true;
  }
  condition.notify_all();
  worker.join();
}

template <class F> void Thread::run(F &&f) {
  auto task = std::make_shared<std::packaged_task<void()>>(
      std::bind(std::forward<F>(f)));
  results.emplace(task->get_future());
  {
    std::unique_lock<std::mutex> lock(mutex);
    tasks.emplace([task]() { (*task)(); });
  }
  condition.notify_one();
}

void Thread::synchronize() {
  //std::cout << "[THREAD] SYNCHRONIZE" << std::endl;
  if (results.empty())
    return;
  results.front().get();
  results.pop();
}

class MultiThread {
public:
  MultiThread(size_t threads);
  ~MultiThread();
  template <class F> void run(F &&f);
  void synchronize();

private:
  bool stop;
  std::mutex mutex;
  std::vector<std::thread> workers;
  std ::condition_variable condition;
  std::queue<std::future<void>> results;
  std::queue<std::function<void()>> tasks;
};

inline MultiThread::MultiThread(size_t threads) : stop(false) {
  for (size_t i =0; i< threads; ++i)
    workers.emplace_back(
  [this] {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(this->mutex);
        this->condition.wait(
            lock, [this] { return this->stop || !this->tasks.empty(); });
        if (this->stop && this->tasks.empty())
          return;
        task = std::move(this->tasks.front());
        this->tasks.pop();
      }
      task();
    }
  });
}

inline MultiThread::~MultiThread() {
  {
    std::unique_lock<std::mutex> lock(mutex);
    stop = true;
  }
  condition.notify_all();
  for(std::thread &worker: workers)
    worker.join();
}

template <class F> void MultiThread::run(F &&f) {
  auto task = std::make_shared<std::packaged_task<void()>>(
      std::bind(std::forward<F>(f)));
  results.emplace(task->get_future());
  {
    std::unique_lock<std::mutex> lock(mutex);
    tasks.emplace([task]() { (*task)(); });
  }
  condition.notify_one();
}

void MultiThread::synchronize() {
  //std::cout << "[THREAD] SYNCHRONIZE" << std::endl;
  while (!results.empty())
    {
      results.front().get();
      results.pop();
    }
}
