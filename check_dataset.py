from argparse import ArgumentParser

from tqdm import tqdm

from common.dataset import Dataset
from common.pen.solve import Solver


def read_arguments():
    parser = ArgumentParser()

    env = parser.add_argument_group('Dataset & Evaluation')
    env.set_defaults(curriculum=True, imitation=True, ignore_incorrect=True)
    env.add_argument('--dataset', '-data', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = read_arguments()

    solver = Solver()

    # Load dataset
    dataset = Dataset(args.dataset)
    for _item in tqdm(dataset._whole_items):
        answers, err = solver.solve(_item.equation.to_sympy(_item.info.variables), _item.info.numbers)
        if err:
            print('Exception (%s) occurred in %s' % (str(err), _item.info.item_id))
            continue

        if not solver.check_answer(_item.info.answers, answers):
            print('Answer is not same in %s (%s)\n\tExpected %s\n\tResulted %s' % (_item.info.item_id, _item.info.raw['dataset'],
                                                                              str(_item.info.answers), answers))

    print('Finished for %s' % dataset.get_dataset_name)

    solver.close()
