#pragma once
#include<functional>
#include<memory>
#include<vector>

using namespace std;

class VariableImpl {
public:
	double value;
	double grad;
	function<void()> _backward;
	vector<shared_ptr<VariableImpl>> _parents;
	bool visited;

	VariableImpl(double value);
};