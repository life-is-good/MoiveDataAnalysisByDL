# -*- coding:utf-8 -*-
from collections import OrderedDict
import cPickle as pkl
import sys
import time
import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import textprocessing
import pdb
from gof.opt import optimizer
from dask.array.tests.test_array_core import test_size
from sklearn.cluster.tests.test_k_means import n_samples

datasets = {'my_data': (textprocessing.load_data, textprocessing.prepare_data)}

# 为持久化设计生成器的种子树
SEED = 123
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)

def get_minibatches_idx(n, minibatch_size, shuffle=True):
    #打乱数据并按照minibatch拿到数据
    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def get_dataset(name):
    #获得训练、校验、测试集的数据，准备集的数据
    return datasets[name][0], datasets[name][1]

def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

#dropout层

def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before * 
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj

def _p(pp, name):
    """
    Return string representation
    """
    return '%s_%s' % (pp, name)

def init_params(options):
    #初始化全局的参数而不是lstm的参数
    params = OrderedDict()
    # embedding
    randn = numpy.random.rand(options['n_words'],
                              options['dim_proj'])
    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params

def load_params(path, params):
    #从模型中加载参数
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params

def init_tparams(params):
    #初始化theano的参数
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def get_layer(name):
    """
    Return layer as specified in name
    """
    fns = layers[name]
    return fns

def ortho_weight(ndim):
    """
    Return matrix orthogonal to the weight matrix.
    """
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)

def param_init_lstm(options, params, prefix='lstm'):
    #初始化lstm的参数
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params

#lstm层

def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) + 
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]


# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.

layers = {'lstm': (param_init_lstm, lstm_layer)}

#随机梯度下降算法
def sgd(lr, tparams, grads, x, mask, y, cost):
    
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update

def adadelta(lr, tparams, grads, x, mask, y, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update

def rmsprop(lr, tparams, grads, x, mask, y, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update

#生成模型

def build_model(tparams, options):

    trng = RandomStreams(SEED)

    # 使用dropout
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]
    #首先tparams[‘Wemb’]里面存的是词向量（每个词一个dim_proj，缺省调用了），
    #通过[x.flatten()].reshape延展成了一个3d矩阵
    #（maxlen最长句子的单词数，bath_size 每次扫描的单词数，dim_proj词向量／lstm hidden layer的个数） 
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)
    if options['encoder'] == 'lstm':
        proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost

#使用已经训练好的模型计算新样本的概率

def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples classified' % (n_done, n_samples)

    return probs

#theano的预测函数，标签数据，训练数据，交叉验证之后的数据

def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    #计算错误率
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        preds = f_pred(x, mask)
        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])
    return valid_err

def test_lstm(
    dim_proj=20,
    patience=20,
    max_epochs=30,
    dispFreq=10,
    decay_c=0., 
    lrate=0.1,  
    n_words=100000,
    optimizer=adadelta,  
    encoder='lstm',
    saveto='lstm_mydata_model.npz',
    validFreq=170,
    saveFreq=1110,  
    maxlen=None, 
    batch_size=40,
    valid_batch_size=64,
    dataset='my_data',
    use_dropout=True,
    noise_std=0.,
    reload_model=None,
    test_size=-1,
    ):

    model_options = locals().copy()
    print "模型参数选项", model_options

    load_data, prepare_data = get_dataset(dataset)

    print '加载数据'
    train, valid, test = load_data(n_words=n_words, valid_portion=0.05,maxlen=maxlen)
    if test_size > 0:
        #将测试集合打乱
        idx = numpy.arange(len(test[0]))
        numpy.random.shuffle(idx)
        test = ([test[0][n] for n in idx],[test[1][n] for n in idx])    
    
    ydim = numpy.max(train[1]) + 1
    model_options['ydim'] = ydim

    print '生成模型'
    # 初始化自定义参数，就是传进来的参数
    params = init_params(model_options)
#     if reload_model:
#         load_params('lstm_model.npz', params)
#     初始化theano参数
    tparams = init_tparams(params)

    (use_noise, x, mask,y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    if decay_c > 0:
        decay_c = theano.shared(numpy_floatX(decay_c),name = 'decay_c')
        weight_decay = 0
        weight_decay += (tparams['U']**2).sum()
        weight_decay *= decay_c
        cost += weight_decay
    f_cost = theano.function([x,mask,y],cost,name = 'f_cost')
    grads = tensor.grad(cost,wrt = tparams.values())
    f_grad = theano.function([x,mask,y],grads,name = 'f_grad')
    lr = tensor.scalar(name = 'lr')
    f_grad_shared,f_update = optimizer(lr,tparams,grads,x,mask,y,cost)
     
    # 按照批次分开训练集，测试集，验证集,需要先shuffle,千万不能忘了
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print "%d 训练集" % len(train[0])
    print "%d 校验集" % len(valid[0])
    print "%d 测试集" % len(test[0])

    print "开始训练过程"
    history_errs = []
    best_p = None
    bad_count = 0
    
    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size

    uidx = 0
    estop = False #是否earlystop
    start_time = time.clock()

    try:
        for eidx in xrange(max_epochs):
            ismodel = 0
            n_samples = 0
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle = True)
            for _,train_index in kf:
                uidx += 1
                use_noise.set_value(1.)
                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index]
                x,mask,y = prepare_data(x,y)
                n_samples += x.shape[1]
                cost = f_grad_shared(x,mask,y)
                f_update(lrate)
                
                if numpy.isnan(cost) or numpy.isinf(cost):
                    print "检测到Nan值 "
                    return 1.,1.,1.
                
                if numpy.mod(uidx,dispFreq) == 0:
                    print 'epoch',eidx,'update',uidx,'cost',cost
                
#                 if saveto and numpy.mod(uidx,saveFreq) == 0:
#                     print "正在保存模型"
#                     if best_p is not None:
#                         params = best_p
#                     else:
#                         params = unzip(tparams)
#                     numpy.savez(saveto,history_errs = history_errs,**params)
#                     pkl.dump(model_options, open("%s.pkl" %saveto,"wb"),-1)
#                     print "模型保存成功"
                
                if numpy.mod(uidx,validFreq) == 0:
                    use_noise.set_value(0.)
                    train_err = pred_error(f_pred, prepare_data, train, kf)
                    valid_err = pred_error(f_pred, prepare_data, valid,kf_valid)
                    test_err = pred_error(f_pred, prepare_data, test, kf_test)
 
                    history_errs.append([valid_err, test_err])
 
                    if (uidx == 0 or valid_err <= numpy.array(history_errs)[:,0].min()):
                        best_p = unzip(tparams)
                        bad_counter = 0
 
                    print '训练集错误率 ', train_err, '验证集错误率 ', valid_err,'测试集错误率 ', test_err
 
                    if (len(history_errs) > patience and
                        valid_err >= numpy.array(history_errs)[:-patience,0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break
                    #错误率小于0.1就可以保存了。
                    if test_err < 0.10:
                        print "开始保存模型"
                        if best_p is not None:
                            params = best_p
                        else:
                            params = unzip(tparams)
#                         if test_err < min([his_errs[1] for his_errs in history_errs]):
                        ismodel = 1
#                             history_errs = test_err
                        numpy.savez(saveto,history_errs = history_errs,**params)
                        pkl.dump(model_options, open("%s.pkl" %saveto,"wb"),-1)
                        print "模型保存完毕"
                        break
            print '有 %d 样本' % n_samples
 
            if estop:
                break
                        
    except KeyboardInterrupt:
        print "训练意外终止"
    end_time = time.clock()
    print "训练花费时间："
    print end_time-start_time
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)
        
    params = init_params(model_options)
    if reload_model and ismodel == 1:
        load_params('lstm_model.npz', params)
#     初始化theano参数
    tparams = init_tparams(params)    
    
    use_noise.set_value(0.)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error(f_pred, prepare_data, test, kf_test)
    print "最终结果："
    print '训练集错误率 ：', train_err, '校验集错误率 ：', valid_err, '测试集错误率： ', test_err
    if saveto:
        numpy.savez(saveto, train_err=train_err,valid_err=valid_err, test_err=test_err,history_errs=history_errs, **best_p)
    return train_err, valid_err, test_err

# def train_lstm(
#     dim_proj=128,  # word embeding 的维数和隐藏层的维数，使用默认值（将一个词转换为一个向量的过程）
#     patience=10,  # 用于earlystop，如果10轮迭代的误差没有降低，就进行earlystop
#     max_epochs=10000,  # 迭代的轮数
#     dispFreq=10,  # 每经过10轮训练就显示训练验证和测试误差
#     decay_c=0.,  # 参数u的正则权重，u为隐藏层ht到输出层的参数
#     lrate=0.0001,  # sgd用的学习率
#     n_words=10000,  # 词典大小，用于数据预处理部分，将词用该词在词典中的ID表示，超过10000的用1表示，仅仅用于数据，
#     optimizer=adadelta,  # 优化方法，代码提供了sgd,adadelta和rmsprop三种方法，采用了adadelta
#     encoder='lstm',  # 一个标识符，可以去掉，但是必须是lstm.
#     saveto='lstm_model.npz',  # 保存最好模型的文件，保存训练误差，验证误差和测试误差等等
#     validFreq=370,  # 验证频率
#     saveFreq=1110,  # 保存频率
#     maxlen=100,  # 序列的最大长度，超出长度的数据被抛弃，见数据处理部分
#     batch_size=16,  # 训练的batch大小
#     valid_batch_size=64,  # 验证集用的*batch大小
#     dataset='imdb',
#     noise_std=0.,
#     use_dropout=True,  # 控制dropout，不用dropout的话运行速度较快，但是效果不好，dropout不太好解释，以一定的概率随机丢弃模型中的一些节点，这样可以综合多种模型的结果，进行投票。需要自行补充deeplearning的知识
#                        # This frequently need a bigger model.
#     reload_model=None,  # 加载模型参数的文件，用于已训练好的模型，或保存的中间结果
#     test_size=-1,  # 测试集大小，如果为正，就只用这么多测试样本
# ):
# 
#     # Model options
#     model_options = locals().copy()
#     print "model options", model_options
# 
#     load_data, prepare_data = get_dataset(dataset)
# 
#     print 'Loading data'
#     train, valid, test = load_data(n_words=n_words, valid_portion=0.05,
#                                    maxlen=maxlen)
#     if test_size > 0:
#         # The test set is sorted by size, but we want to keep random
#         # size example.  So we must select a random selection of the
#         # examples.
#         idx = numpy.arange(len(test[0]))
#         numpy.random.shuffle(idx)
#         idx = idx[:test_size]
#         test = ([test[0][n] for n in idx], [test[1][n] for n in idx])
# 
#     ydim = numpy.max(train[1]) + 1
# 
#     model_options['ydim'] = ydim
# 
#     print 'Building model'
#     # This create the initial parameters as numpy ndarrays.
#     # Dict name (string) -> numpy ndarray
#     params = init_params(model_options)
# 
#     if reload_model:
#         load_params('lstm_model.npz', params)
# 
#     # This create Theano Shared Variable from the parameters.
#     # Dict name (string) -> Theano Tensor Shared Variable
#     # params and tparams have different copy of the weights.
#     tparams = init_tparams(params)
# 
#     # use_noise is for dropout
#     (use_noise, x, mask,
#      y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)
#     
# 
#     if decay_c > 0.:
#         decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
#         weight_decay = 0.
#         weight_decay += (tparams['U'] ** 2).sum()
#         weight_decay *= decay_c
#         cost += weight_decay
# 
#     f_cost = theano.function([x, mask, y], cost, name='f_cost')
# 
#     grads = tensor.grad(cost, wrt=tparams.values())
#     f_grad = theano.function([x, mask, y], grads, name='f_grad')
# 
#     lr = tensor.scalar(name='lr')
#     f_grad_shared, f_update = optimizer(lr, tparams, grads,
#                                         x, mask, y, cost)
# 
#     print 'Optimization'
# 
#     kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
#     kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)
# 
#     print "%d train examples" % len(train[0])
#     print "%d valid examples" % len(valid[0])
#     print "%d test examples" % len(test[0])
# 
#     history_errs = []
#     best_p = None
#     bad_count = 0
# 
#     if validFreq == -1:
#         validFreq = len(train[0]) / batch_size
#     if saveFreq == -1:
#         saveFreq = len(train[0]) / batch_size
# 
#     uidx = 0  # the number of update done
#     estop = False  # early stop
#     start_time = time.clock()
#     try:
#         for eidx in xrange(max_epochs):
#             n_samples = 0
# 
#             # Get new shuffled index for the training set.
#             kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)
# 
#             for _, train_index in kf:
#                 uidx += 1
#                 use_noise.set_value(1.)
# 
#                 # Select the random examples for this minibatch
#                 y = [train[1][t] for t in train_index]
#                 x = [train[0][t]for t in train_index]
# 
#                 # Get the data in numpy.ndarray format
#                 # This swap the axis!
#                 # Return something of shape (minibatch maxlen, n samples)
#                 x, mask, y = prepare_data(x, y)
#                 n_samples += x.shape[1]
# 
#                 cost = f_grad_shared(x, mask, y)
#                 f_update(lrate)
# 
#                 if numpy.isnan(cost) or numpy.isinf(cost):
#                     print 'NaN detected'
#                     return 1., 1., 1.
# 
#                 if numpy.mod(uidx, dispFreq) == 0:
#                     print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost
# 
#                 if saveto and numpy.mod(uidx, saveFreq) == 0:
#                     print 'Saving...',
# 
#                     if best_p is not None:
#                         params = best_p
#                     else:
#                         params = unzip(tparams)
#                     numpy.savez(saveto, history_errs=history_errs, **params)
#                     pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
#                     print 'Done'
# 
#                 if numpy.mod(uidx, validFreq) == 0:
#                     use_noise.set_value(0.)
#                     train_err = pred_error(f_pred, prepare_data, train, kf)
#                     valid_err = pred_error(f_pred, prepare_data, valid,
#                                            kf_valid)
#                     test_err = pred_error(f_pred, prepare_data, test, kf_test)
# 
#                     history_errs.append([valid_err, test_err])
# 
#                     if (uidx == 0 or
#                         valid_err <= numpy.array(history_errs)[:,
#                                                                0].min()):
# 
#                         best_p = unzip(tparams)
#                         bad_counter = 0
# 
#                     print ('Train ', train_err, 'Valid ', valid_err,
#                            'Test ', test_err)
# 
#                     if (len(history_errs) > patience and
#                         valid_err >= numpy.array(history_errs)[:-patience,
#                                                                0].min()):
#                         bad_counter += 1
#                         if bad_counter > patience:
#                             print 'Early Stop!'
#                             estop = True
#                             break
# 
#             print 'Seen %d samples' % n_samples
# 
#             if estop:
#                 break
# 
#     except KeyboardInterrupt:
#         print "Training interupted"
# 
#     end_time = time.clock()
#     if best_p is not None:
#         zipp(best_p, tparams)
#     else:
#         best_p = unzip(tparams)
# 
#     use_noise.set_value(0.)
#     kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
#     # 使用theano里面的方法进行预测
#     train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
#     valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
#     test_err = pred_error(f_pred, prepare_data, test, kf_test)
#     # 最后的准确性
#     print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err
#     if saveto:
#         numpy.savez(saveto, train_err=train_err,
#                     valid_err=valid_err, test_err=test_err,
#                     history_errs=history_errs, **best_p)
#     print 'The code run for %d epochs, with %f sec/epochs' % (
#         (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
#     print >> sys.stderr, ('Training took %.1fs' % 
#                           (end_time - start_time))
#     return train_err, valid_err, test_err
if __name__ == '__main__':
    # 可以赋值一些训练参数之类的.
#     train_lstm(
#         max_epochs=1,
#         test_size=500,
#     )
    # test the model
    start_time = time.time()
    print start_time
    test_lstm(dataset='my_data')
    end_time = time.time()
    print end_time - start_time

