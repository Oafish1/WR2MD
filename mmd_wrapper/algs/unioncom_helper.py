from .source.unioncom import *

# 'Change:' comments are changes for this wrapper
class unioncom_helper(UnionCom):
    """
    Performs UnionCom and records statistics during runtime

    Parameters
    ----------
    **kwargs: kwargs
        Any parameters to use while running the specified method

    Returns
    -------
    History object with various entries returned depending on
    the parameters chosen
    tsne:
        'loss', 'loss_align'
    MultiOmics:
        'corr_alpha'
    All Methods:
        'corr_err', 'corr_iteration', 'iteration'
    """

    def __init__(self, **kwargs):
        # Change: Logging
        self._history = {}
        super().__init__(**kwargs)

    def Prime_Dual(self, dist, dx=None, dy=None):
        """
        prime dual combined with Adam algorithm to find the local optimal soluation
        """

        print("use device:", self.device)

        if self.integration_type == "MultiOmics":
            Kx = dist[0]
            Ky = dist[1]
            N = int(np.maximum(len(Kx), len(Ky)))
            Kx = Kx / N
            Ky = Ky / N
            Kx = torch.from_numpy(Kx).float().to(self.device)
            Ky = torch.from_numpy(Ky).float().to(self.device)
            a = np.sqrt(dy/dx)
            m = np.shape(Kx)[0]
            n = np.shape(Ky)[0]

        else:
            m = np.shape(dist)[0]
            n = np.shape(dist)[1]
            a=1
            dist = torch.from_numpy(dist).float().to(self.device)

        F = np.zeros((m,n))
        F = torch.from_numpy(F).float().to(self.device)
        Im = torch.ones((m,1)).float().to(self.device)
        In = torch.ones((n,1)).float().to(self.device)
        Lambda = torch.zeros((n,1)).float().to(self.device)
        Mu = torch.zeros((m,1)).float().to(self.device)
        S = torch.zeros((n,1)).float().to(self.device)

        pho1 = 0.9
        pho2 = 0.999
        delta = 10e-8
        Fst_moment = torch.zeros((m,n)).float().to(self.device)
        Snd_moment = torch.zeros((m,n)).float().to(self.device)

        i=0
        while(i<self.epoch_pd):

            ### compute gradient

            # tmp = Kx - torch.mm(F, torch.mm(Ky, torch.t(F)))
            # w_tmp = -4*torch.abs(tmp) * torch.sign(tmp)
            # grad1 = torch.mm(w_tmp, torch.mm(F, torch.t(Ky)))

            # tmp = torch.mm(torch.t(F), torch.mm(a*Kx, F)) - Ky
            # w_tmp = 4*torch.abs(tmp) * torch.sign(tmp)
            # grad2 = torch.mm(Kx, torch.mm(F, torch.t(w_tmp)))

            if self.integration_type == "MultiOmics":
                grad = 4*torch.mm(F, torch.mm(Ky, torch.mm(torch.t(F), torch.mm(F, Ky)))) \
                - 4*a*torch.mm(Kx, torch.mm(F,Ky)) + torch.mm(Mu, torch.t(In)) \
                + torch.mm(Im, torch.t(Lambda)) + self.rho*(torch.mm(F, torch.mm(In, torch.t(In))) - torch.mm(Im, torch.t(In)) \
                + torch.mm(Im, torch.mm(torch.t(Im), F)) + torch.mm(Im, torch.t(S-In)))
            else:
                grad = dist + torch.mm(Im, torch.t(Lambda)) + self.rho*(torch.mm(F, torch.mm(In, torch.t(In))) - torch.mm(Im, torch.t(In)) \
                + torch.mm(Im, torch.mm(torch.t(Im), F)) + torch.mm(Im, torch.t(S-In)))
            # print(dist)
            ### adam momentum
            i += 1
            Fst_moment = pho1*Fst_moment + (1-pho1)*grad
            Snd_moment = pho2*Snd_moment + (1-pho2)*grad*grad
            hat_Fst_moment = Fst_moment/(1-np.power(pho1,i))
            hat_Snd_moment = Snd_moment/(1-np.power(pho2,i))
            grad = hat_Fst_moment/(torch.sqrt(hat_Snd_moment)+delta)
            F_tmp = F - grad
            F_tmp[F_tmp<0]=0

            ### update
            F = (1-self.epsilon)*F + self.epsilon*F_tmp

            ### update slack variable
            grad_s = Lambda + self.rho*(torch.mm(torch.t(F), Im) - In + S)
            s_tmp = S - grad_s
            s_tmp[s_tmp<0]=0
            S = (1-self.epsilon)*S + self.epsilon*s_tmp

            ### update dual variables
            Mu = Mu + self.epsilon*(torch.mm(F,In) - Im)
            Lambda = Lambda + self.epsilon*(torch.mm(torch.t(F), Im) - In + S)

            #### if scaling factor changes too fast, we can delay the update
            if self.integration_type == "MultiOmics":
                if i>=self.delay:
                    a = torch.trace(torch.mm(Kx, torch.mm(torch.mm(F, Ky), torch.t(F)))) / \
                    torch.trace(torch.mm(Kx, Kx))

            if (i+1) % self.log_pd == 0:
                if self.integration_type == "MultiOmics":
                    norm2 = torch.norm(a*Kx - torch.mm(torch.mm(F, Ky), torch.t(F)))
                    # Change: Logging
                    for k in ['corr_iteration', 'corr_err', 'corr_alpha']:
                        if k not in self._history:
                            self._history[k] = []
                    self._history['corr_iteration'].append(i)
                    self._history['corr_err'].append(norm2.data.item())
                    self._history['corr_alpha'].append(a)
                    print("epoch:[{:d}/{:d}] err:{:.4f} alpha:{:.4f}".format(i+1, self.epoch_pd, norm2.data.item(), a))
                else:
                    norm2 = torch.norm(dist*F)
                    # Change: Logging
                    for k in ['corr_iteration', 'corr_err']:
                        if k not in self._history:
                            self._history[k] = []
                    self._history['corr_iteration'].append(i)
                    self._history['corr_err'].append(norm2.data.item())
                    print("epoch:[{:d}/{:d}] err:{:.4f}".format(i+1, self.epoch_pd, norm2.data.item()))

        F = F.cpu().numpy()
        return F

    def project_barycentric(self, dataset, match_result):
        print("---------------------------------")
        print("Begin finding the embedded space")
        integrated_data = []
        for i in range(len(dataset)-1):
            integrated_data.append(np.matmul(match_result[i], dataset[-1]))
        integrated_data.append(dataset[-1])
        print("Done")
        return integrated_data

    def project_tsne(self, dataset, pairs_x, pairs_y, P_joint):
        print("---------------------------------")
        print("Begin finding the embedded space")

        net = model(self.col, self.output_dim)
        Project_DNN = init_model(net, self.device, restore=None)

        optimizer = optim.RMSprop(Project_DNN.parameters(), lr=self.lr)
        c_mse = nn.MSELoss()
        Project_DNN.train()

        dataset_num = len(dataset)

        for i in range(dataset_num):
            P_joint[i] = torch.from_numpy(P_joint[i]).float().to(self.device)
            dataset[i] = torch.from_numpy(dataset[i]).float().to(self.device)

        for epoch in range(self.epoch_DNN):
            len_dataloader = int(np.max(self.row)/self.batch_size)
            if len_dataloader == 0:
                len_dataloader = 1
                self.batch_size = np.max(self.row)
            for step in range(len_dataloader):
                KL_loss = []
                for i in range(dataset_num):
                    random_batch = np.random.randint(0, self.row[i], self.batch_size)
                    data = dataset[i][random_batch]
                    P_tmp = torch.zeros([self.batch_size, self.batch_size]).to(self.device)
                    for j in range(self.batch_size):
                        P_tmp[j] = P_joint[i][random_batch[j], random_batch]
                    P_tmp = P_tmp / torch.sum(P_tmp)
                    low_dim_data = Project_DNN(data, i)
                    Q_joint = Q_tsne(low_dim_data)

                    ## loss of structure preserving
                    KL_loss.append(torch.sum(P_tmp * torch.log(P_tmp / Q_joint)))

        		## loss of structure matching
                feature_loss = np.array(0)
                feature_loss = torch.from_numpy(feature_loss).to(self.device).float()
                for i in range(dataset_num-1):

                    low_dim = Project_DNN(dataset[i][pairs_x[i]], i)
                    low_dim_biggest_dataset = Project_DNN(dataset[dataset_num-1][pairs_y[i]], len(dataset)-1)
                    feature_loss += c_mse(low_dim, low_dim_biggest_dataset)
                    # min_norm = torch.min(torch.norm(low_dim), torch.norm(low_dim_biggest_dataset))
                    # feature_loss += torch.abs(torch.norm(low_dim) - torch.norm(low_dim_biggest_dataset))/min_norm

                loss = self.beta * feature_loss
                for i in range(dataset_num):
                    loss += KL_loss[i]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch+1) % self.log_DNN == 0:
                # Change: Logging
                for k in ['iteration', 'loss', 'loss_align']:
                    if k not in self._history:
                        self._history[k] = []
                self._history['iteration'].append(epoch)
                self._history['loss'].append(loss.data.item())
                self._history['loss_align'].append(feature_loss.data.item())
                print("epoch:[{:d}/{}]: loss:{:4f}, align_loss:{:4f}".format(epoch+1, \
                    self.epoch_DNN, loss.data.item(), feature_loss.data.item()))

        integrated_data = []
        for i in range(dataset_num):
            integrated_data.append(Project_DNN(dataset[i], i))
            integrated_data[i] = integrated_data[i].detach().cpu().numpy()
        print("Done")
        return integrated_data


# if __name__ == '__main__':

    # data1 = np.loadtxt("./Seurat_scRNA/CTRL_PCA.txt")
    # data2 = np.loadtxt("./Seurat_scRNA/STIM_PCA.txt")
    # type1 = np.loadtxt("./Seurat_scRNA/CTRL_type.txt")
    # type2 = np.loadtxt("./Seurat_scRNA/STIM_type.txt")

    ### batch correction for HSC data
    # data1 = np.loadtxt("./hsc/domain1.txt")
    # data2 = np.loadtxt("./hsc/domain2.txt")
    # type1 = np.loadtxt("./hsc/type1.txt")
    # type2 = np.loadtxt("./hsc/type2.txt")

    ### UnionCom simulation
    # data1 = np.loadtxt("./simu1/domain1.txt")
    # data2 = np.loadtxt("./simu1/domain2.txt")
    # type1 = np.loadtxt("./simu1/type1.txt")
    # type2 = np.loadtxt("./simu1/type2.txt")
    #-------------------------------------------------------

    ### MMD-MA simulation
    # data1 = np.loadtxt("./MMD/s1_mapped1.txt")
    # data2 = np.loadtxt("./MMD/s1_mapped2.txt")
    # type1 = np.loadtxt("./MMD/s1_type1.txt")
    # type2 = np.loadtxt("./MMD/s1_type2.txt")
    #-------------------------------------------------------

    ### scGEM data
    # data1 = np.loadtxt("./scGEM/GeneExpression.txt")
    # data2 = np.loadtxt("./scGEM/DNAmethylation.txt")
    # type1 = np.loadtxt("./scGEM/type1.txt")
    # type2 = np.loadtxt("./scGEM/type2.txt")
    #-------------------------------------------------------

    ### scNMT data
    # data1 = np.loadtxt("./scNMT/Paccessibility_300.txt")
    # data2 = np.loadtxt("./scNMT/Pmethylation_300.txt")
    # data3 = np.loadtxt("./scNMT/RNA_300.txt")
    # type1 = np.loadtxt("./scNMT/type1.txt")
    # type2 = np.loadtxt("./scNMT/type2.txt")
    # type3 = np.loadtxt("./scNMT/type3.txt")
    # not_connected, connect_element, index = Maximum_connected_subgraph(data3, 40)
    # if not_connected:
    # 	data3 = data3[connect_element[index]]
    # 	type3 = type3[connect_element[index]]
    # min_max_scaler = preprocessing.MinMaxScaler()
    # data3 = min_max_scaler.fit_transform(data3)
    # print(np.shape(data3))
    #-------------------------------------------------------

    # print(np.shape(data1))
    # print(np.shape(data2))

    ### integrate two datasets
    # type1 = type1.astype(np.int)
    # type2 = type2.astype(np.int)
    # uc = UnionCom(distance_mode='geodesic', project_mode='tsne', integration_type="MultiOmics", batch_size=100)
    # integrated_data = uc.fit_transform(dataset=[data1,data2])
    # uc.test_LabelTA(integrated_data, [type1,type2])
    # uc.Visualize([data1,data2], integrated_data, [type1,type2], mode='PCA')

    ## integrate three datasets
    # type1 = type1.astype(np.int)
    # type2 = type2.astype(np.int)
    # type3 = type3.astype(np.int)
    # datatype = [type1,type2,type3]
    # uc = UnionCom()

    # inte = uc.fit_transform([data1,data2,data3])
    # uc.test_LabelTA(inte, [type1,type2,type3])
    # uc.Visualize([data1,data2,data3], inte, datatype, mode='UMAP')
