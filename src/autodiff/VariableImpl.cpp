#include "../../includes/autodiff/VariableImpl.h"
#include<vector>
#include<memory>
#include<functional>

VariableImpl::VariableImpl(long double value) {
    this->value = value;
    grad = 0.0;
    _backward = []() {};
    visited = false;
}