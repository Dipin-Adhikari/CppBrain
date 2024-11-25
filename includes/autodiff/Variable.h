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
	Variable(long double value);
	Variable();
	long double getValue();
	long double getGrad();
	void clearGraph();
	Variable operator +(const Variable& other);
	Variable operator -(const Variable& other);
	Variable operator *(const Variable& other);
	Variable operator /(const Variable& other);
	friend Variable operator +(const long double n, const Variable& other);
	friend Variable operator -(const long double n, const Variable& other);
	friend Variable operator *(const long double n, const Variable& other);
	friend Variable operator /(const long double n, const Variable& other);
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
