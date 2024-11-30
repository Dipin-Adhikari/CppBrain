#pragma once
#include<functional>
#include<vector>
#include<memory>
#include"VariableImpl.h"

class Variable {
private:
	shared_ptr<VariableImpl> impl;
	explicit Variable(shared_ptr<VariableImpl> impl);
public:
	Variable(double value);
	Variable();
	double getValue();
	double getGrad();
	void clearGraph();
	Variable operator +(const Variable& other);
	Variable operator -(const Variable& other);
	Variable operator *(const Variable& other);
	Variable operator /(const Variable& other);
	friend Variable operator +(const double n, const Variable& other);
	friend Variable operator -(const double n, const Variable& other);
	friend Variable operator *(const double n, const Variable& other);
	friend Variable operator /(const double n, const Variable& other);
	Variable operator-();

	Variable power(const Variable& other);
	Variable sine();
	Variable cosine();
	Variable tangent();
	Variable exponential();
	Variable logarithm();
	Variable absolute();

	void backward();
};
