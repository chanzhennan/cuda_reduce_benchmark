

3 ^
    8

    3 ^
    2 * 3 ^ 2 * 3 ^ 2 * 3 ^
    2

    3 ^
    15 i = 3

           3 ^
           8 * 3 ^ 7 i = 2

                         3 ^
                         4 3 ^ 4 * 2 ^ 4 * 2 ^
                         3 i = 1

                               3 ^
                               2 3 ^ 2 3 ^ 2 3 ^ 2 3 ^ 2 3 ^ 2 3 ^
                               2 i = 0

                                     15 8 +
                                     7 8 + 6 +
                                     1

                                     13 8 +
                                     4 +
                                     1

                                     70

                                     1 2 4 8 16 32 64

                                     64 +
                                     4 +
                                     2

                                     float
                                     pow(float x, int 35) {
  if (n == 1) return x;

  float* pv = (float*)malloc(sizeof(float) * n);
  pv[0] = x * x;
  float res = 1.f;
  for (int i = 1; i < n; i >>= 2) {
    pv[i] = pv[i - 1] * pv[i - 1];
  }

  for (int i = 0; i < n; i++) {
  }
}
