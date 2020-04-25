# coding=utf-8
import json
import sys
import os
import string
import numpy as np
import time
import glob
import zipfile
import shutil


def glob_matching(fn, fmt):
    matched_fns = list(glob.iglob('submit/**/submission.csv', recursive=True))
    if len(matched_fns) == 0:
        raise Exception("You submitted a {} file, but we didn't find submission.csv in it. Please check your submission.".format(fmt))
    if len(matched_fns) > 1:
        raise Exception("You submitted a {} file, but there are more than one files named submission.csv in it. Please check your submission.".format(fmt))
    return matched_fns[0]


def read_submission(submit_path, reference, k=5):
    # check whether the path of submitted file exists
    if not os.path.exists(submit_path):
        raise Exception("The submission file is not found!")

    # evaluate a zip file
    if os.path.isdir("submit"):
        shutil.rmtree("submit")
    if submit_path.endswith('.zip'):
        try:
            with zipfile.ZipFile(submit_path, "r") as zip_data:
                zip_data.extractall("submit")
                zip_data.close()
        except:
            raise Exception('The submitted zip file is corrputed! Please check your submission.')
        real_submit_path = glob_matching('submission.csv', 'zip')
    # evaluate a csv file
    else:
        real_submit_path = submit_path

    submission_dict = {}
    ref_qids = set(reference.keys())

    with open(real_submit_path) as fin:
        for line in fin:
            line = line.strip()
            records = [elem.strip() for elem in line.split(',')]
            if records[0] not in ref_qids:
                continue
            qid = records[0]
            # check whether there are K products for each query
            if len(records[1:]) != k:
                raise Exception('Query-id {} has wrong number of predicted product-ids! Require {}, but {} founded.'.format(qid, k, len(records[1:])))
            # check whether there exists an empty prediction for any query
            if any([len(r) == 0 for r in records[1:]]):
                raise Exception('Query-id {} has an empty prediction at rank {}! Pleace check again!'.format(qid, records[1:].index("") + 1))            
            # check whether there exist an invalid prediction for any query
            for rank, r in enumerate(records[1:]):
                if not all([char in string.digits for char in r]):
                    raise Exception('Query-id {} has an invalid prediction product-id \"{}\" at rank {}'.format(qid, r, rank + 1))
            # check whether there are duplicate predicted products for a single query
            if len(set(records[1:])) != k:
                raise Exception('Query-id {} has duplicate products in your prediction. Pleace check again!'.format(qid))
            submission_dict[qid] = records[1:] # here we save the list of string
    
    # check if any query is missing in the submission
    pred_qids = set(submission_dict.keys())
    nopred_qids = ref_qids - pred_qids
    if len(nopred_qids) != 0:
        raise Exception('The following query-ids have no prediction in your submission, please check again: {}'.format(", ".join(nopred_qids)))

    return submission_dict
    

# compute dcg@k for a single sample
def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(3, r.size + 2)))
    return 0.


# compute ndcg@k (dcg@k / idcg@k) for a single sample
def get_ndcg(r, ref, k):
    dcg_max = dcg_at_k(ref, k)
    if not dcg_max:
        return 0.
    dcg = dcg_at_k(r, k)
    return dcg / dcg_max


def dump_2_json(info, path):
    with open(path, 'w') as output_json_file:
        json.dump(info, output_json_file)


def report_error_msg(detail, showMsg, out_p):
    error_dict=dict()
    error_dict['errorDetail']=detail
    error_dict['errorMsg']=showMsg
    error_dict['score']=0
    error_dict['scoreJson']={}
    error_dict['success']=False
    dump_2_json(error_dict,out_p)


def report_score(score, out_p):
    result = dict()
    result['success']=True
    result['score'] = score
    result['scoreJson'] = {'score': score}
    dump_2_json(result,out_p)


if __name__=="__main__":
    # the path of answer json file (eg. valid_answer.json)
    standard_path = sys.argv[1]
    # the path of prediction file (csv or zip)
    submit_path = sys.argv[2]
    # the score will be dumped into this output json file
    out_path = sys.argv[3]

    print("Read standard from %s" % standard_path)
    print("Read user submit file from %s" % submit_path)

    try:
        # read ground-truth
        reference = json.load(open(standard_path))
        
        # read predictions
        k = 5
        predictions = read_submission(submit_path, reference, k)

        # compute score for each query
        score_sum = 0.
        for qid in reference.keys():
            ground_truth_ids = set([str(pid) for pid in reference[qid]])
            ref_vec = [1.0] * len(ground_truth_ids)
            pred_vec = [1.0 if pid in ground_truth_ids else 0.0 for pid in predictions[qid]]
            score_sum += get_ndcg(pred_vec, ref_vec, k)
        # the higher score, the better
        score = score_sum / len(reference)
        report_score(score, out_path)
        print("The evaluation finished successfully.")
    except Exception as e:
        report_error_msg(e.args[0], e.args[0], out_path)
        print("The evaluation failed: {}".format(e.args[0]))