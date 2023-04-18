import datetime
import os


def write_log(w, args):
    file_name = '../logs/' + datetime.date.today().strftime('%m%d') + \
                f"_{args.num_part}_{args.method}.log"
    if not os.path.exists('../logs/'):
        os.mkdir('../logs/')
    t0 = datetime.datetime.now().strftime('%H:%M:%S')
    info = "{} : {}".format(t0, w)
    print(info)
    with open(file_name, 'a') as f:
        f.write(info + '\n')
