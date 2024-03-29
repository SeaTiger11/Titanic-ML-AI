int test[2] = {1, 8};

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

string code = "int test[2] = {1, " + to_string(++test[1]) + "};";
string line = "";

int main() {
    fstream file("rewrite.cpp");
    getline(file, line);
    while(getline(file, line)) {
        code += "\n" + line;
    }

    file.close();
    file.open("rewrite.cpp", fstream::out | fstream::trunc);
    file << code;
    file.close();
}