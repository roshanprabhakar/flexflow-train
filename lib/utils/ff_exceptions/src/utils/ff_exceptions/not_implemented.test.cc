#include "utils/ff_exceptions/not_implemented.h"
#include "utils/testing.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("not_implemented::not_implemented()") {
    not_implemented n{};
  }
}
