#include "lattice.hpp"


HoppingMatrix makeTriangle()
{
    HoppingMatrix hopping(3, 3);
    hopping.set(0, 1, 1.0);
    hopping.set(1, 0, 1.0);
    hopping.set(0, 2, 1.0);
    hopping.set(2, 0, 1.0);
    hopping.set(1, 2, 1.0);
    hopping.set(2, 1, 1.0);
    return hopping;
}
