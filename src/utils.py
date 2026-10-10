import re
import subprocess
import time
import os
import shutil
from multiprocessing import Pool, TimeoutError
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import logging
import pickle

__author__ = 'Yuxing Xu'
SCRIPT_DIR_PATH = os.path.split(os.path.realpath(__file__))[0]


def pickle_dump(save_object, output_pickle_file):
    try:
        OUT = open(output_pickle_file, 'wb')
        pickle.dump(save_object, OUT)
        OUT.close()
        return output_pickle_file
    except:
        raise ValueError("Failed to write %s" % output_pickle_file)


def pickle_load(input_pickle_file):
    try:
        TEMP = open(input_pickle_file, 'rb')
        output_object = pickle.load(TEMP)
        TEMP.close()
        return output_object
    except:
        raise ValueError("Failed to open %s" % input_pickle_file)


def time_now():
    time_tmp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    return time_tmp


def log_print(print_string):
    time_tmp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print("%s\t\t\t%s" % (time_tmp, print_string))


def logging_init(program_name, log_file=None, log_level=logging.DEBUG, console_level=logging.ERROR):
    # create logger with 'program_name'
    logger = logging.getLogger(program_name)
    logger.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if not log_file is None:
        # create file handler which logs even debug messages
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)

    return logger


def printer_list(list_input, sep="\t", head="", wrap_num=None):
    """
    make a list to a string with a given sep chara and head
    :param list_input: list you give me
    :param sep: Split character like "," or "\t"
    :param head: head for the output string
    :return: string
    """
    printer = head
    num = 0

    for i in list_input:
        printer = printer + str(i) + sep
        num = num + 1
        if wrap_num:
            if num % wrap_num == 0:
                printer = printer + "\n"

    printer = printer.rstrip(sep)
    return printer


def rmdir(dir_name):
    if os.path.exists(dir_name):
        if os.path.isdir(dir_name):
            shutil.rmtree(dir_name)
        else:
            os.remove(dir_name)


def is_path(path):
    if os.path.isfile(path):
        return True
    elif os.path.isdir(path):
        return True
    else:
        return False


def if_file_in_directory(file_name, directory):
    file_name = get_file_name(file_name)
    return os.path.isfile(os.path.join(directory, file_name))


def get_file_name(give_path):
    return os.path.basename(give_path)


def get_file_dir(file_path):
    abs_file_path = os.path.abspath(file_path)
    return os.path.dirname(abs_file_path)


def copy_file(source_path, target_path, keep=False):
    source_path = os.path.abspath(source_path)
    target_path = os.path.abspath(target_path)

    if not os.path.exists(source_path):
        raise EnvironmentError("Can't find %s" % source_path)

    if os.path.isdir(source_path):
        new_path = shutil.copytree(source_path, os.path.join(
            target_path, get_file_name(source_path)))
    elif os.path.isfile(source_path):
        if os.path.isdir(target_path):
            target_path = os.path.join(target_path, get_file_name(source_path))

        if keep and os.path.exists(target_path):
            new_path = target_path
        else:
            new_path = shutil.copy(source_path, target_path)

    return new_path


def ln_file(source_path, target_path):

    if os.path.isdir(target_path):
        target_path = target_path + "/" + get_file_name(source_path)

    os.symlink(os.path.abspath(source_path), target_path)

    return target_path


def move_file(source_path, target_path, keep=False):
    source_path = os.path.abspath(source_path)
    target_path = os.path.abspath(target_path)

    new_path = shutil.move(source_path, target_path)

    return new_path


def merge_file(input_file_list, output_file):
    with open(output_file, 'w') as f:
        for input_file in input_file_list:
            fr = open(input_file, 'r').read()
            if len(fr) > 0 and fr[-1] != '\n':
                fr = fr+'\n'
            f.write(fr)


def gunzip_file(raw_file):
    raw_file = os.path.abspath(raw_file)

    cmd_string = "gunzip " + raw_file
    cmd_run(cmd_string, silence=True)

    gunzip_file = re.sub(".gz$", "", raw_file)

    return gunzip_file


def gzip_file(raw_file):
    raw_file = os.path.abspath(raw_file)

    cmd_string = "gzip " + raw_file
    cmd_run(cmd_string, silence=True)

    gzip_file = raw_file + ".gz"

    return gzip_file


def md5sum_check(file_name, original_md5):
    import hashlib

    if not os.path.exists(file_name):
        return False

    with open(file_name, "rb") as file_to_check:
        data = file_to_check.read()
        md5_returned = hashlib.md5(data).hexdigest()

    if original_md5 == md5_returned:
        return True
    else:
        return False


def remove_file_name_suffix(file_name, subsuffix_level=0):
    """
    remove the suffix for a file name
    :param file_name: give me a file name, like: "/home/xuyuxing/file.txt" or "~/work/file.txt" or just "file.txt"
    :param subsuffix_level: sometimes a file can have more than one suffix, like "~/work/file.txt.gz",
           how many you want to remove, 0 meanings remove all suffix
    :return: file without suffix, will keep dir_name as a path, to get base file name: os.path.basename(file_name)
    """

    file_name = os.path.abspath(file_name)
    dir_name = os.path.dirname(file_name)
    base_name = os.path.basename(file_name)
    base_name_split = base_name.split(".")
    if subsuffix_level == 0:
        subsuffix_level = len(base_name_split) - 1
    elif subsuffix_level > len(base_name_split) - 1:
        raise ValueError("There are not enough suffix level to remove")

    return dir_name + "/" + printer_list(base_name.split(".")[:-subsuffix_level], ".")


def md5sum_maker(file_name):
    import hashlib

    if not os.path.exists(file_name):
        FileNotFoundError("No such file or directory: %s" % file_name)

    with open(file_name, "rb") as file_to_check:
        data = file_to_check.read()
        md5_returned = hashlib.md5(data).hexdigest()

    return md5_returned


def cmd_run(cmd_string, cwd=None, retry_max=5, silence=True, log_file=None):
    module_logger = logging_init("cmd_run", log_file)
    module_logger.info("Calling a bash cmd with retry_max %d: %s" %
                       (retry_max, cmd_string))
    if not silence:
        print("Running " + str(retry_max) + " " + cmd_string)
    p = subprocess.Popen(cmd_string, shell=True,
                         stderr=subprocess.PIPE, stdout=subprocess.PIPE, cwd=cwd)
    output, error = p.communicate()
    if not silence:
        print(error.decode())
    returncode = p.poll()
    module_logger.info("Finished bash cmd with returncode as %d" % returncode)
    if returncode == 1:
        if retry_max > 1:
            retry_max = retry_max - 1
            cmd_run(cmd_string, cwd=cwd, retry_max=retry_max)
    del module_logger.handlers[:]

    output = output.decode()
    error = error.decode()

    return (not returncode, output, error)


def multiprocess_running(f, args_dict, pool_num, log_file=None, silence=False, timeout=None, only_keep_output=False):
    """
    :param f: function name
    :param args_dict: a dict, value is every args in a tuple, key is the job_id
    :param pool_num: process num of running at same time
    :param silence:
    :return:

    example

    def f(x, y):
        time.sleep(1)
        return x * y

    args_list = list(zip(range(1,200),range(2,201)))
    pool_num = 5

    multiprocess_running(f, args_list, pool_num, log_file='/lustre/home/xuyuxing/Work/Other/saif/Meth/tmp/log')

    """
    num_tasks = len(list(args_dict))

    module_log = logging_init(f.__name__, log_file)
    if not silence:
        print('args_list have %d object and pool_num is %d' %
              (num_tasks, pool_num))
    module_log.info('running with mulitprocess')
    module_log.info('args_list have %d object and pool_num is %d' %
                    (num_tasks, pool_num))

    p_dict = {}

    args_id_list = list(args_dict.keys())
    args_list = [args_dict[i] for i in args_id_list]

    if len(args_dict) > 0:
        f_with_para = partial(get_more_para, f)

        start_time = time.time()
        if not silence:
            print(time_now() + '\tBegin: ')
        module_log.info('Begin: ')

        if timeout is None:
            with Pool(processes=pool_num) as pool:
                # for i, output in enumerate(pool.imap_unordered(f_with_para, args_list, chunksize=1)):
                for i, output in enumerate(pool.imap(f_with_para, args_list, chunksize=1)):
                    job_id = args_id_list[i]

                    if only_keep_output:
                        p_dict[job_id] = output
                    else:
                        p_dict[job_id] = {
                            'args': args_list[i],
                            'output': output,
                            'error': None
                        }

                    # print(i)
                    round_time = time.time()
                    if round_time - start_time > 5:
                        if not silence:
                            print(time_now() + '\t%d/%d %.2f%% parsed' %
                                  (i, num_tasks, i / num_tasks * 100))
                        module_log.info('%d/%d %.2f%% parsed' %
                                        (i, num_tasks, i / num_tasks * 100))
                        start_time = round_time
                    # module_log.info('%d/%d %.2f%% parsed' % (i, num_tasks, i / num_tasks * 100))
        else:
            abortable_func = partial(
                abortable_worker, f_with_para, timeout=timeout)

            # with Pool(processes=pool_num, maxtasksperchild=1) as pool:
            with Pool(processes=pool_num) as pool:
                # for i, output in enumerate(pool.imap_unordered(f_with_para, args_list, chunksize=1)):
                it = pool.imap(abortable_func, args_list, chunksize=1)
                i = -1
                while 1:
                    i += 1
                    try:
                        output = it.next()
                        if args_id_list:
                            job_id = args_id_list[i]
                        else:
                            job_id = 'ID_' + str(i)

                        if output == "Aborting due to timeout":
                            if only_keep_output:
                                p_dict[job_id] = output
                            else:
                                p_dict[job_id] = {
                                    'args': args_list[i],
                                    'output': None,
                                    'error': "timeout"
                                }
                        else:
                            if only_keep_output:
                                p_dict[job_id] = output
                            else:
                                p_dict[job_id] = {
                                    'args': args_list[i],
                                    'output': output,
                                    'error': None
                                }

                        round_time = time.time()
                        if round_time - start_time > 5:
                            if not silence:
                                print(time_now() + '\t%d/%d %.2f%% parsed' %
                                      (i, num_tasks, i / num_tasks * 100))
                            module_log.info('%d/%d %.2f%% parsed' %
                                            (i, num_tasks, i / num_tasks * 100))
                            start_time = round_time

                    except StopIteration:
                        break

        module_log.info('%d/%d %.2f%% parsed' %
                        (i, num_tasks, i / num_tasks * 100))
        module_log.info('All args_list task finished')

        if not silence:
            print(time_now() + '\t%d/%d %.2f%% parsed' %
                  (i, num_tasks, i / num_tasks * 100))
            print(time_now() + '\tAll args_list task finished')

    del module_log.handlers[:]

    return p_dict


def get_more_para(f, para_tuple):
    return f(*para_tuple)


def abortable_worker(func, *args, **kwargs):
    timeout = kwargs.get('timeout', None)
    p = ThreadPool(1)
    res = p.apply_async(func, args=args)
    try:
        out = res.get(timeout)  # Wait timeout seconds for func to complete.
        return out
    except TimeoutError:
        return "Aborting due to timeout"
        raise


if __name__ == "__main__":
    pass
