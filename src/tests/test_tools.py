from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp-name local_test --datasets mnist --approach finetuning" \
                       " --network LeNet --num-tasks 10 --stop-at-task 3 --seed 1 --batch-size 64" \
                       " --nepochs 2" \
                       " --num-workers 0"


def test_log_grad_norm():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --log-grad-norm"
    run_main_and_assert(args_line)


def test_umap_latent_visualization():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --umap-latent"
    run_main_and_assert(args_line)


def test_last_head_analysis():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --wu-nepochs 2"
    args_line += " --last-head-analysis"
    run_main_and_assert(args_line)
