#include "../../includes/autodiff/Variable.h"
#include<vector>
#include<memory>
#include<functional>
#include"../../includes/autodiff/VariableImpl.h"
#include<iostream>
#include<unordered_set>
#include<algorithm>
#include <math.h>


Variable::Variable(shared_ptr<VariableImpl> impl) {
    this->impl = move(impl);
}




Variable::Variable(long double value) {
    impl = make_shared<VariableImpl>(value);
}

Variable::Variable() {
    impl = make_shared<VariableImpl>(0.0);
}

long double Variable::getValue() {
    return impl->value;
}

long double Variable::getGrad() {
    return impl->grad;
}

void Variable::clearGraph() {
    if (impl) {
        impl->grad = 0.0;
        impl->visited = false;
        impl->_parents.clear();
        impl->_backward = []() {};
    }
}

Variable Variable::operator+(const Variable& other) {
    auto out = make_shared<VariableImpl>(impl->value + other.impl->value);

    out->_parents.push_back(impl);
    out->_parents.push_back(other.impl);
    out->_backward = [out, this_impl = impl, other_impl = other.impl]() {
        this_impl->grad += out->grad;
        other_impl->grad += out->grad;
    };

    return Variable(out);
}

Variable operator +(const long double n, const Variable& other) {
    return Variable(n) + other;
}


Variable Variable::operator-(const Variable& other) {
    auto out = make_shared<VariableImpl>(impl->value - other.impl->value);

    out->_parents.push_back(impl);
    out->_parents.push_back(other.impl);
    out->_backward = [out, this_impl = impl, other_impl = other.impl]() {
        this_impl->grad += out->grad;
        other_impl->grad -= out->grad;
    };

    return Variable(out);
}

Variable operator -(const long double n, const Variable& other) {
    return Variable(n) - other;
}

Variable Variable::operator -() {
    auto out = make_shared<VariableImpl>(-impl->value);
    out->_parents = { impl };
    out->_backward = [out, this_impl = impl]() {
        this_impl->grad -= out->grad;
    };
    return Variable(out);
}

Variable Variable::operator*(const Variable& other) {
    auto out = make_shared<VariableImpl>(impl->value * other.impl->value);

    out->_parents.push_back(impl);
    out->_parents.push_back(other.impl);
    out->_backward = [out, this_impl = impl, other_impl = other.impl]() {
        this_impl->grad += other_impl->value * out->grad;
        other_impl->grad += this_impl->value * out->grad;
    };

    return Variable(out);
}

Variable operator *(const long double n, const Variable& other) {
    return Variable(n) * other;
}

Variable Variable::operator/(const Variable& other) {
    auto out = make_shared<VariableImpl>(impl->value / other.impl->value);

    out->_parents.push_back(impl);
    out->_parents.push_back(other.impl);
    out->_backward = [out, this_impl = impl, other_impl = other.impl]() {
        this_impl->grad += (1 / other_impl->value) * out->grad;
        other_impl->grad -= (this_impl->value / (other_impl->value * other_impl->value)) * out->grad;
    };

    return Variable(out);
}

Variable operator /(const long double n, const Variable& other) {
    return Variable(n) / other;
}



Variable Variable::power(const Variable& other) {
    auto out = make_shared<VariableImpl>(pow(impl->value, other.impl->value));
    out->_parents.push_back(impl);
    out->_parents.push_back(other.impl);
    out->_backward = [out, this_impl = impl, other_impl = other.impl](){
        this_impl->grad += other_impl->value * pow(this_impl->value, (other_impl->value - 1)) * out->grad;
        other_impl->grad += pow(this_impl->value, other_impl->value) * log(this_impl->value) * out->grad;
    };
    return Variable(out);
}

Variable Variable::sine() {
    auto out = make_shared<VariableImpl>(sin(impl->value));

    out->_parents.push_back(impl);
    out->_backward = [out, this_impl = impl]() {
        this_impl->grad += cos(this_impl->value) * out->grad;
    };

    return Variable(out);
}


Variable Variable::cosine() {
    auto out = make_shared<VariableImpl>(cos(impl->value));

    out->_parents.push_back(impl);
    out->_backward = [out, this_impl = impl]() {
        this_impl->grad -= sin(this_impl->value) * out->grad;
    };

    return Variable(out);
}

Variable Variable::tangent() {
    auto out = make_shared<VariableImpl>(tan(impl->value));

    out->_parents.push_back(impl);
    out->_backward = [out, this_impl = impl]() {
        this_impl->grad += (1 / pow(cos(this_impl->value), 2)) * out->grad;
    };

    return Variable(out);
}

Variable Variable::exponential() {
    auto out = make_shared<VariableImpl>(exp(impl->value));

    out->_parents.push_back(impl);
    out->_backward = [out, this_impl = impl, out_impl = out]() {
        this_impl->grad += out_impl->value * out->grad;
    };

    return Variable(out);
}

Variable Variable::logarithm() {
    auto out = make_shared<VariableImpl>(log(impl->value));

    out->_parents.push_back(impl);
    out->_backward = [out, this_impl = impl, out_impl = out]() {
        if (this_impl->value > 0) {
            this_impl->grad += (1 / this_impl->value) * out->grad;
        }
        else {
            this_impl->grad += (1 / (1e-08)) * out->grad;
        }
    };
    return Variable(out);
}

Variable Variable::absolute() {
    auto out = make_shared<VariableImpl>(abs(impl->value));

    out->_parents.push_back(impl);
    out->_backward = [out, this_impl = impl, out_impl = out]() {
        this_impl->grad += (this_impl->value / abs(this_impl->value)) * out->grad;
    };
    return Variable(out);
}




void Variable::backward() {
    vector<shared_ptr<VariableImpl>> topo;
    unordered_set<shared_ptr<VariableImpl>> visited;

    function<void(const shared_ptr<VariableImpl>&)> buildTopo = [&](const shared_ptr<VariableImpl>& f) {
        if (visited.insert(f).second) {
            for (const auto& parent : f->_parents) {
                buildTopo(parent);
            }
            topo.push_back(f);
        }
    };

    buildTopo(impl);

    impl->grad = 1.0;

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->_backward();
    }
}




