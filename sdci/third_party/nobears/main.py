"""
From https://github.com/howchihlee/BNGPU/blob/master/BNGPU/NOBEARS.py.
"""
import tensorflow as tf
import numpy as np
from scipy.optimize import lsq_linear
from scipy.stats.stats import spearmanr

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_resource_variables()


def W_reg_init(X):
    n, d = X.shape
    rho, p = spearmanr(X)
    np.fill_diagonal(rho, 0)

    W_init = np.zeros((d, d))
    for i in range(d):
        ind = np.argsort(rho[:, i])[::-1][: min([int(n * 0.5), 30])]
        Xpar = X[:, ind]
        Xpar += 0.01 * np.random.randn(*Xpar.shape)
        yc = X[:, i]
        wpar = lsq_linear(Xpar, yc)["x"]

        W_init[ind, i] = wpar
    W_init = W_init
    return W_init


def power_iteration_tf(W, power_iteration_rounds=1):
    ## W nxn tensor

    ## modified from https://github.com/google/compare_gan/blob/master/compare_gan/src/gans/ops.py
    # Persisted approximation of first left singular vector of matrix `w`.

    ## right eigenvector,
    u_var = tf.compat.v1.get_variable(
        W.name.replace(":", "") + "/u_var",
        shape=(W.shape[0], 1),
        dtype=W.dtype,
        initializer=tf.compat.v1.random_uniform_initializer(),
        trainable=False,
    )

    ## left eigenvector,
    v_var = tf.compat.v1.get_variable(
        W.name.replace(":", "") + "/v_var",
        shape=(W.shape[0], 1),
        dtype=W.dtype,
        initializer=tf.compat.v1.random_uniform_initializer(),
        trainable=False,
    )

    u = u_var
    v = v_var

    for _ in range(power_iteration_rounds):
        v = tf.nn.l2_normalize(
            tf.matmul(W, v, transpose_a=True), axis=None, epsilon=1e-12
        )
        u = tf.nn.l2_normalize(tf.matmul(W, u), axis=None, epsilon=1e-12)

        # Update persisted approximation.
    with tf.control_dependencies([tf.compat.v1.assign(u_var, u, name="update_u")]):
        u = tf.identity(u)

    with tf.control_dependencies([tf.compat.v1.assign(v_var, v, name="update_v")]):
        v = tf.identity(v)

    u = tf.stop_gradient(u)
    v = tf.stop_gradient(v)

    # Largest singular value of `w`.
    norm_value = tf.matmul(tf.matmul(v, W, transpose_a=True), u) / tf.reduce_sum(v * u)
    norm_value.shape.assert_is_fully_defined()
    norm_value.shape.assert_is_compatible_with([1, 1])

    return u_var, v_var, u, v, norm_value


class NoBearsTF:
    def __init__(
        self,
        beta1=0.05,
        beta2=0.001,
        alpha_init=0.01,
        rho_init=1.0,
        poly_degree=3,
        l1_ratio=0.5,
    ):
        self.alpha = alpha_init
        self.rho = rho_init
        self.beta1 = beta1
        self.beta2 = beta2
        self.poly_degree = poly_degree
        self.l1_ratio = l1_ratio

    def roar(self):
        print("Our time is short!")

    def model_init_train(
        self, sess, init_iter=200, init_global_step=1e-2, noise_level=0.1
    ):
        for t1 in range(init_iter):
            feed_dict = {
                self.graph_nodes["alpha"]: self.alpha,
                self.graph_nodes["rho"]: self.rho,
                self.graph_nodes["opt_step"]: init_global_step
                / np.sqrt(1.0 + 0.1 * t1),
                self.graph_nodes["beta1"]: self.beta1,
                self.graph_nodes["beta2"]: self.beta2,
                self.graph_nodes["noise_level"]: noise_level,
            }
            sess.run(self.graph_nodes["init_train_op"], feed_dict=feed_dict)
        return

    def model_train(
        self,
        sess,
        outer_iter=200,
        inner_iter=100,
        init_global_step=1e-2,
        noise_level=0.1,
        eval_fun=None,
    ):
        ave_loss_reg = -1
        ave_loss_min = np.inf

        sess.run(self.graph_nodes["data_iterator_init"])
        for t in range(outer_iter):
            sess.run(self.graph_nodes["reset_opt"])
            for t1 in range(inner_iter):
                feed_dict = {
                    self.graph_nodes["alpha"]: self.alpha,
                    self.graph_nodes["rho"]: self.rho,
                    self.graph_nodes["opt_step"]: init_global_step / np.sqrt(1.0 + t1),
                    self.graph_nodes["beta1"]: self.beta1,
                    self.graph_nodes["beta2"]: self.beta2,
                    self.graph_nodes["noise_level"]: noise_level,
                }

                _, v0, h_val = sess.run(
                    [
                        self.graph_nodes["train_op"],
                        self.graph_nodes["loss_regress"],
                        self.graph_nodes["loss_penalty"],
                    ],
                    feed_dict=feed_dict,
                )

            sess.run(self.graph_nodes["moving_average_op"])

            if ave_loss_reg < 0:
                ave_loss_reg = v0
            else:
                ave_loss_reg = 0.1 * ave_loss_reg + 0.9 * v0
            if ave_loss_min > ave_loss_reg:
                ave_loss_min = ave_loss_reg
            elif ave_loss_reg > (1.5 * ave_loss_min) and h_val < 0.005:
                return

            self.alpha += self.rho * h_val
            self.rho = min(1e8, 1.1 * self.rho)

    def construct_graph(self, X, W_init=None, buffer_size=100, batch_size=None):
        n, d = X.shape
        if batch_size is None:
            batch_size = n // 2

        tf.compat.v1.reset_default_graph()

        X_datatf = tf.constant(X.astype("float32"), name="sem_data")  ## n x d
        X_dataset = tf.data.Dataset.from_tensor_slices(X_datatf)
        X_iterator = tf.compat.v1.data.make_initializable_iterator(
            X_dataset.shuffle(buffer_size=buffer_size).batch(batch_size).repeat()
        )
        X_tf = X_iterator.get_next()
        X_tf = tf.constant(X.astype("float32"), name="sem_data")  ## n x d

        batch_size_tf = tf.cast(tf.shape(X_tf)[0], tf.float32)

        if W_init is None:
            W_tf = tf.compat.v1.get_variable(
                "W", [d, d], initializer=tf.compat.v1.random_uniform_initializer()
            )
        else:
            W_tf = tf.compat.v1.get_variable("W", initializer=W_init)

        W_tf = tf.linalg.set_diag(W_tf, tf.zeros(d))

        ## ema
        ema = tf.train.ExponentialMovingAverage(decay=0.975)
        ema_op = ema.apply([W_tf])
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, ema_op)
        W_ema = ema.average(W_tf)

        noise_level_tf = tf.compat.v1.placeholder(tf.float32, name="noise_level")

        X0_tf = X_tf + noise_level_tf * tf.random.normal(tf.shape(X_tf), stddev=1.0)

        ## setup objective function

        bk0_tf = tf.compat.v1.get_variable(
            "bk_poly0",
            shape=(1, d),
            initializer=tf.compat.v1.random_uniform_initializer(),
        )
        X1_tf = bk0_tf

        regression_vars = [bk0_tf]
        for k in range(1, self.poly_degree + 1):
            sig_tf = tf.compat.v1.get_variable(
                "sig_poly%d" % k,
                shape=(1, d),
                initializer=tf.compat.v1.random_uniform_initializer(),
            )
            regression_vars.append(sig_tf)

            if k == 1:
                X1_tf = X1_tf + (X0_tf) * sig_tf
            else:
                X1_tf = X1_tf + (X0_tf**k) * sig_tf

        X2_tf = tf.matmul(X1_tf, W_tf)

        loss_xw = tf.reduce_sum((X2_tf - X_tf) ** 2) / (2.0 * batch_size_tf)

        W2 = W_tf * W_tf
        W2p = W2 + 1e-6

        vr_var, vl_var, vr, vl, h_tf = power_iteration_tf(W2p, power_iteration_rounds=5)

        loss_penalty = tf.reduce_sum(h_tf)

        rho_tf = tf.compat.v1.placeholder(tf.float32, name="rho")
        beta1_tf = tf.compat.v1.placeholder(tf.float32, name="beta1")
        beta2_tf = tf.compat.v1.placeholder(tf.float32, name="beta2")
        d_lrd = tf.compat.v1.placeholder(tf.float32, name="opt_step")
        alpha_tf = tf.compat.v1.placeholder(tf.float32, name="alpha")

        l1_ratio = self.l1_ratio
        W_abs = tf.abs(W_tf)
        reg0 = tf.reduce_sum(W_abs)
        reg1 = tf.reduce_sum(W2) * 0.5

        if l1_ratio < 1e-5:
            loss_reg = beta1_tf * reg0
        elif l1_ratio > 0.99999:
            loss_reg = beta1_tf * reg1
        else:
            loss_reg = beta1_tf * ((1.0 - l1_ratio) * reg1 + l1_ratio * reg0)

        for w in regression_vars:
            loss_reg = loss_reg + beta2_tf * tf.reduce_sum(w**2)

        loss_init = loss_xw + loss_reg
        loss_obj = (
            loss_init + (rho_tf / 2.0) * (loss_penalty**2) + alpha_tf * loss_penalty
        )

        # optimizers
        optim = tf.compat.v1.train.AdamOptimizer(d_lrd, beta1=0.9)
        sn_ops = [
            op
            for op in tf.compat.v1.get_default_graph().get_operations()
            if "update_u" in op.name or "update_v" in op.name
        ]

        with tf.control_dependencies(sn_ops):
            train_op = optim.minimize(loss_obj)

        optim_init = tf.compat.v1.train.AdamOptimizer(d_lrd, beta1=0.9)
        with tf.control_dependencies(sn_ops):
            init_train_op = optim_init.minimize(loss_init)

        reset_optimizer_op = tf.compat.v1.variables_initializer(optim.variables())

        model_init = tf.compat.v1.global_variables_initializer()

        self.graph_nodes = {
            "sem_data": X_tf,
            "weight": W_tf,
            "weight_ema": W_ema,
            "rho": rho_tf,
            "loss_penalty": loss_penalty,
            "beta1": beta1_tf,
            "beta2": beta2_tf,
            "opt_step": d_lrd,
            "alpha": alpha_tf,
            "noise_level": noise_level_tf,
            "loss_regress": loss_xw,
            "loss_obj": loss_obj,
            "train_op": train_op,
            "reset_opt": reset_optimizer_op,
            "update_uv": sn_ops,
            "init_vars": model_init,
            "data_iterator_init": X_iterator.initializer,
            "moving_average_op": ema_op,
            "init_train_op": init_train_op,
        }

        return self.graph_nodes
