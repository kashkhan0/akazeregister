#include <sys/stat.h>
#include <sys/types.h>



bool isfile(const std::string &name) {
  ifstream f(name.c_str());
  return f.good();
}


int imuriToNum(string imuri){ 
 vector<string> tokens;
 split2(imuri,'/', tokens);
 if( tokens[0] == "") tokens.erase(begin(tokens));
 int nt=tokens.size(); // number of tokens
 return atoi(tokens[nt-2].substr(1,2).c_str())*999 + atoi(tokens[nt-1].substr(4,4).c_str());
}
