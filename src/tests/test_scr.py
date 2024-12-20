from tests import run_main_and_assert
import pytest
from argparse import ArgumentError

FAST_LOCAL_TEST_ARGS = "--exp-name local_test --datasets cifar100_icarl" \
                       " --network resnet32 --num-tasks 10 --stop-at-task 2 --seed 1 --batch-size 128" \
                       " --nepochs 1" \
                       " --num-workers 0" \
                       " --approach scr" \
                       " --extra-aug simclr_cifar"


def test_scr_with_exemplars_nmc():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200"
    args_line += " --classifier nmc"
    run_main_and_assert(args_line)


def test_scr_knn():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200 --classifier knn"
    run_main_and_assert(args_line)


def test_scr_cont_eval():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200 --cont-eval"
    args_line += " --classifier nmc"
    run_main_and_assert(args_line)


def test_scr_supcon():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200"
    args_line += " --loss-func supcon"
    args_line += " --classifier nmc"
    run_main_and_assert(args_line)


def test_scr_linear_projector():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200"
    args_line += " --projector-type linear"
    args_line += " --classifier nmc"
    run_main_and_assert(args_line)


def test_scr_without_exemplars():
    with pytest.raises(ValueError):
        run_main_and_assert(FAST_LOCAL_TEST_ARGS)


def test_scr_linear_clf():
    with pytest.raises(ValueError):
        args_line = FAST_LOCAL_TEST_ARGS
        args_line += " --num-exemplars 200"
        args_line += " --classifier linear"
        run_main_and_assert(args_line)
