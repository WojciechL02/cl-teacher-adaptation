from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp-name local_test --datasets mnist" \
                       " --network LeNet --num-tasks 5 --seed 1 --batch-size 32" \
                       " --nepochs 2" \
                       " --num-workers 0" \
                       " --approach ssil"


def test_ssil():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200"
    run_main_and_assert(args_line)


def test_ssil_nmc():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200 --classifier nmc"
    run_main_and_assert(args_line)


def test_ssil_knn():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200 --classifier knn"
    run_main_and_assert(args_line)


def test_ssil_cont_eval():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200 --cont-eval"
    run_main_and_assert(args_line)
