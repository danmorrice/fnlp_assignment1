from autodrive_embed import carefulBind

import sys

def trim_and_warn(name, max_len, s):
    s = [x for x in s.split(" ") if x]
    if len(s) > max_len:
        print("\nWarning - truncating output of %s: your answer has %i words but the limit is %i" % (
        name, len(s), max_len), file=sys.stderr)
    return " ".join(s[:max_len-1])

def extract_answers(gdict, errlog):
  global _me
  _me=None
  globals().update(gdict)
  try:
    from s2061990 import logistic_regression_model
    _me=logistic_regression_model.classifier_obj
  except:
    pass
  (cerrs,ans)=carefulBind(
        [('lm_stats', '''[lm._N,
                             lm.prob('h', 't'),
                             lm.prob('u', 'q'),
                             lm.prob('z', 'q'),
                             lm.prob('j', ('<s>',), True),
                             lm.prob('</s>', 'e', True)]'''),
        ('dev_tweets_preds', "dev_tweets_preds"),
        ('answer_short_1_3', "answer_short_1_3"),
        ('answer_short_1_4', "answer_short_1_4"),
        ('answer_short_1_5', "answer_short_1_5"),
        ("top10_ents", "top10_ents"),
        ("bottom10_ents", "bottom10_ents"),
        ('answer_essay_question', "answer_essay_question"),
        ("naive_bayes_vocab_size", "len(naive_bayes.vocab)"),
        ("naive_bayes_prior", "naive_bayes.prior"),
        ("naive_bayes_likelihood", '''[naive_bayes.likelihood.get(c,{}).get(f,FAILED) for c,f in [("V",("v", "rose")),
("V",("p", "of")),
("N",("p", "of")),
("N",("n2", "609")),
("V",("n2", "609")),
("V",("n1", "million")),
("N",("n1", "million"))]]'''),
        ("naive_bayes_posterior", '''[naive_bayes.prob_classify([("v", "took")]),
                                  naive_bayes.prob_classify([("v", "took"), ("n1", "advantage")]),
                                  naive_bayes.prob_classify([("p", "to")]),
                                  naive_bayes.prob_classify([("n1", "dsfjghdkfjgh"), ("p", "to")]),
                                  naive_bayes.prob_classify(
                                      [("v", "values"), ("n1", "plan"), ("p", "at"), ("n2", "billion")])
                                  ]'''),
        ("naive_bayes_classify", '''[naive_bayes.classify([("v", "took")]),
                              naive_bayes.classify([("v", "took"), ("n1", "advantage")])]'''),
        ("naive_bayes_acc", "naive_bayes_acc"),
        ('answer_open_question_2_2', "answer_open_question_2_2"),
        ("lr_predictions", '"".join(logistic_regression_model.classify(d) for (d, gold) in dev_features)'),
         ("best10_features", '[(_me._weights[fid],_me._encoding.describe(fid)) for fid in _me.most_informative_features(10)]'),
        ('answer_open_question_2_3', "answer_open_question_2_3"),
     ], globals(), errlog)

  return ans, cerrs
