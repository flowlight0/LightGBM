#ifndef LIGHTGBM_OBJECTIVE_FAIR_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_FAIR_OBJECTIVE_HPP_

#include <LightGBM/objective_function.h>
#include <cstdio>

namespace LightGBM {
  
class LogL1loss: public ObjectiveFunction {
public:
  explicit LogL1loss(const ObjectiveConfig&) {
  }

  ~LogL1loss() {
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();
  }

  void GetGradients(const score_t* score, score_t* gradients,
                    score_t* hessians) const override {
    double fair_constant = 0.7;
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        double x = (score[i] - label_[i]);
        gradients[i] = fair_constant * x / (abs(x) + fair_constant);
        hessians[i] = fair_constant * fair_constant / ((abs(x) + fair_constant) * (abs(x) + fair_constant));
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        double x = (score[i] - label_[i]);
        gradients[i] = weights_[i] * fair_constant * x / (abs(x) + fair_constant);
        hessians[i] = weights_[i] * fair_constant * fair_constant / ((abs(x) + fair_constant) * (abs(x) + fair_constant));
      }
    }
  }

  const char* GetName() const override {
    return "logmae";
  }

private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const float* label_;
  /*! \brief Pointer of weights */
  const float* weights_;
};

}  // namespace LightGBM
#endif   // LightGBM_OBJECTIVE_FAIR_OBJECTIVE_HPP_
