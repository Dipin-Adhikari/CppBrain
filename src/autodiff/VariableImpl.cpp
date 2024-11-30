#include "../../includes/autodiff/VariableImpl.h"
#include<vector>
#include<memory>
#include<functional>

VariableImpl::VariableImpl(double value) {
    this->value = value;
    grad = 0.0;
    _backward = []() {};
    visited = false;
}