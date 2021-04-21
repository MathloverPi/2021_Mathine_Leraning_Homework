import numpy as np
import torch
import matplotlib.pyplot as plt  # 绘图库
class Samples():
    """
    生成数据
    """
    def __init__(self, bound,point_num=100,train_num = 80):
        """
        生成数据
        """
        self.up = bound[1]
        self.low = bound[0]
        self.point_num = point_num
        self.train_num= train_num
        self.test_num= point_num-train_num
        self.all_x = np.arange(self.low,self.up,(self.up-self.low)/self.point_num)#.reshape(self.point_num,1) 
        self.noisy = np.random.normal(loc=0,scale=np.sqrt(5),size= self.point_num)#.reshape(self.point_num,1) 
        self.all_y = 1.4*self.all_x+0.9+self.noisy
        self.split_data()
    def plot_point(self):
        plt.scatter(self.train_x, self.train_y, alpha=0.6,label='Train Point')  # 绘制散点图,透明度为0.6（这样颜色浅一点,比较好看）
        plt.scatter(self.test_x, self.test_y, alpha=0.6,label='Test Point')  # 绘制散点图,透明度为0.6（这样颜色浅一点,比较好看）
        plt.legend()
        plt.show()
    def split_data(self):
        self.train_index = np.array(self.getRandomIndex())
        # 再讲test_index从总的index中减去就得到了train_index
        self.test_index = np.delete(np.arange(self.point_num), self.train_index)
        self.train_x = self.all_x[self.train_index]
        self.train_y = self.all_y[self.train_index]
        self.test_x = self.all_x[self.test_index]
        self.test_y = self.all_y[self.test_index]
    def getRandomIndex(self):
	    # 索引范围为[0, n), 随机选x个不重复
        index = np.random.choice(np.arange(self.point_num),size=self.train_num,replace=False)
        return index

        return 0

class LeastSquares(object):
    def __init__(self, samples):
        """
       最小二乘法
        """
        self.train_x = samples.train_x
        self.train_y = samples.train_y
        self.test_x = samples.test_x
        self.test_y = samples.test_y
        # self.w = []
        self.test_loss = []
        self.train_loss = []
        self.train_num = samples.train_num
        self.test_num = samples.test_num
        # self.loss = samples.data
    def run(self):
        X_T = np.vstack((np.ones(len(self.train_x)),self.train_x))
        X = X_T.T
        Y = self.train_y.reshape(self.train_num,1)
        # print(X.T*X)
        self.w = np.dot(np.dot((np.linalg.inv(np.dot(X.T,X))),X.T),Y)
        self.y_train_pre = self.w[1]*self.train_x+self.w[0]
        self.y_test_pre = self.w[1]*self.test_x+self.w[0]
        self.calculate_loss()
        print(self.w)
        print("LeastSquares Train Loss")
        print(self.train_loss)
        print("LeastSquares Test Loss")
        print(self.test_loss)
        self.plot_point()
        # print(self.w)
        return 0
    def calculate_loss(self):
        self.test_loss = sum((self.y_test_pre - self.test_y) ** 2)/self.test_num
        self.train_loss = sum((self.y_train_pre - self.train_y) ** 2)/self.train_num
        return 0
    def plot_point(self):
        plt.scatter(self.train_x, self.train_y, alpha=0.6,label='Train Point')  # 绘制散点图,透明度为0.6（这样颜色浅一点,比较好看）
        plt.scatter(self.train_x, self.y_train_pre, alpha=0.6,label='Predict Train Point')  # 绘制散点图,透明度为0.6（这样颜色浅一点,比较好看）

        plt.scatter(self.test_x, self.test_y, alpha=0.6,label='Test Point')  # 绘制散点图,透明度为0.6（这样颜色浅一点,比较好看）
        plt.scatter(self.test_x, self.y_test_pre, alpha=0.6,label='Predict Test Point')  # 绘制散点图,透明度为0.6（这样颜色浅一点,比较好看）
        plt.legend()
        plt.show()
class Regressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.ones(1)*1.3)
        self.b = torch.nn.Parameter(torch.ones(1)*0.8)

    def forward(self, x):
        return self.a * x + self.b

class Huber(object):
    def __init__(self, samples):
        """
        Huber方法
        """
        self.train_x = samples.train_x
        self.train_y = samples.train_y
        self.test_x = samples.test_x
        self.test_y = samples.test_y
        self.m = Regressor()
        self.optimizer = torch.optim.Adam(self.m.parameters(), lr=1e-4)
        # self.w = []
        self.test_loss = []
        self.train_loss = []
        self.train_num = samples.train_num
        self.test_num = samples.test_num
        # self.loss = samples.data
    def run(self):
        self.w = self.train(200)
        print(self.w) 
        self.y_train_pre = self.w[1]*self.train_x+self.w[0]
        self.y_test_pre = self.w[1]*self.test_x+self.w[0]
        self.calculate_loss()
        print("LeastSquares Train Loss")
        print(self.train_loss)
        print("LeastSquares Test Loss")
        print(self.test_loss)
        self.plot_point()
        return 0
    def huber_loss(self,y, y_pred, sigma=0.1):
        r = (y - y_pred).abs()
        loss = (r[r <= sigma]).pow(2).mean()
        loss += (r[r > sigma]).mean() * sigma - sigma**2/2
        return loss
    def train(self, n_step=4000):
        for step in range(n_step):
            x_ = torch.FloatTensor(self.train_x)
            y_ = torch.FloatTensor(self.train_y)

            y_pred = self.m(x_)
            self.optimizer.zero_grad()
            loss = self.huber_loss(y_, y_pred)
            loss.backward()
            self.optimizer.step()
        return self.m.b.data.item(), self.m.a.data.item()
    def calculate_loss(self):
        self.test_loss = sum((self.y_test_pre - self.test_y) ** 2)/self.test_num
        self.train_loss = sum((self.y_train_pre - self.train_y) ** 2)/self.train_num
        return 0
    def plot_point(self):
        plt.scatter(self.train_x, self.train_y, alpha=0.6,label='Train Point')  # 绘制散点图,透明度为0.6（这样颜色浅一点,比较好看）
        plt.scatter(self.train_x, self.y_train_pre, alpha=0.6,label='Predict Train Point')  # 绘制散点图,透明度为0.6（这样颜色浅一点,比较好看）
        plt.scatter(self.test_x, self.test_y, alpha=0.6,label='Test Point')  # 绘制散点图,透明度为0.6（这样颜色浅一点,比较好看）
        plt.scatter(self.test_x, self.y_test_pre, alpha=0.6,label='Predict Test Point')  # 绘制散点图,透明度为0.6（这样颜色浅一点,比较好看）
        plt.legend()
        plt.show()


    
    


if __name__ == "__main__":
    samples = Samples(bound=[-100,100],point_num=100,train_num = 80)
    samples.plot_point()
    LeastSquares = LeastSquares(samples)
    LeastSquares.run()
    Huber = Huber(samples)
    Huber.run()
