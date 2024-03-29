#include <iostream>
#include <vector>
#include "knn.h"
using std::vector;
using std::cout;
using std::endl;


int main() {
    Base* obj = new Knn();
    obj->run();
    delete obj;
    return 0;
}