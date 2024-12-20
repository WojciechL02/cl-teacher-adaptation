from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp-name local_test --datasets mnist" \
                       " --network LeNet --num-tasks 2 --seed 1 --batch-size 32" \
                       " --nepochs 2" \
                       " --num-workers 0" \
                       " --approach lwf"


def test_lwf_without_exemplars():
    run_main_and_assert(FAST_LOCAL_TEST_ARGS)


def test_lwf_with_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200"
    run_main_and_assert(args_line)


def test_lwf_with_warmup():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --wu-nepochs 5"
    args_line += " --wu-lr 0.05"
    args_line += " --num-exemplars 200"
    run_main_and_assert(args_line)


def test_lwf_warmup_with_patience():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --wu-nepochs 2 --wu-patience 1"
    run_main_and_assert(args_line)


def test_lwf_warmup_cosine():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --wu-nepochs 2 --wu-scheduler cosine"
    run_main_and_assert(args_line)


def test_lwf_warmup_onecycle():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --wu-nepochs 2 --wu-scheduler onecycle"
    run_main_and_assert(args_line)


def test_lwf_warmup_plateau():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --wu-nepochs 2 --wu-scheduler plateau"
    run_main_and_assert(args_line)


def test_lwf_nmc():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200 --classifier nmc"
    run_main_and_assert(args_line)


def test_lwf_knn():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200 --classifier knn"
    run_main_and_assert(args_line)


def test_lwf_cont_eval():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200 --cont-eval"
    run_main_and_assert(args_line)
