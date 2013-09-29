/*************************************************************************
    > File Name: testvector.cpp
    > Author: onerhao
    > Mail: haodu@hustunique.com
    > Created Time: Sun 24 Mar 2013 03:23:06 PM CST
 ************************************************************************/

#include <vector>
#include <math.h>
#include <stdlib.h>
#include <iostream>

int main(int argc,char **argv)
{
    int i;
    std::vector<int> v;
    for(i=0;i<10;i++)
    {
        v.push_back(i);
    }
    for(i=0;i<v.size();i++)
        std::cout<<v[i]<<",";
    std::cout<<";";
    v.erase(v.begin()+2);
    for(i=0;i<v.size();i++)
        std::cout<<v[i]<<",";
    std::cout<<std::endl;
    if(fabs(1.0-1.1))
        std::cout<<"0.1 "<<false<<std::endl;
    return 0;
}

