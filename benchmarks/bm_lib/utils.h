// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <string>
#include <vector>

namespace ipubm {

// benchmark string helper
std::string strFormat(const char* format, ...);

void genRandom(std::vector<float>& vec);

}  // namespace ipubm
