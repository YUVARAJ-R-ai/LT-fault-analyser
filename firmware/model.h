#pragma once
#include <cstdarg>
namespace Eloquent {
    namespace ML {
        namespace Port {
            class DecisionTree {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        if (x[0] <= 0.2660200670361519) {
                            if (x[0] <= 0.08925898047164083) {
                                return 0;
                            }

                            else {
                                return 1;
                            }
                        }

                        else {
                            if (x[3] <= 1.781329706311226) {
                                return 2;
                            }

                            else {
                                return 3;
                            }
                        }
                    }

                protected:
                };
            }
        }
    }