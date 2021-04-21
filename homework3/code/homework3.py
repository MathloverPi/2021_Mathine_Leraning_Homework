import numpy as np
import torch
import matplotlib.pyplot as plt  # 绘图库
import os
import pandas as pd
import sklearn.linear_model as linear_model
from sklearn.metrics import r2_score

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,Perceptron
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split


class Samples():
    """
    生成数据
    """
    def __init__(self, bound,point_num=5,train_num = 9):
        """
        生成数据
        """
        self.up = bound[1]
        self.low = bound[0]
        self.point_num = point_num
        self.train_num= train_num
        self.test_num= 5
        self.all_x = np.arange(self.low,self.up,(self.up-self.low)/self.point_num)#.reshape(self.point_num,1) 
        self.noisy = np.random.normal(loc=0,scale=np.sqrt(5),size= self.point_num)#.reshape(self.point_num,1) 
        self.all_y = 1.97666667*self.all_x+self.noisy
        self.split_data()
    def plot_point(self):
        plt.scatter(self.train_x, self.train_y, alpha=0.6,label='Train Point')  # 绘制散点图,透明度为0.6（这样颜色浅一点,比较好看）
        plt.scatter(self.test_x, self.test_y, alpha=0.6,label='Test Point')  # 绘制散点图,透明度为0.6（这样颜色浅一点,比较好看）
        plt.legend()
        plt.show()
    def split_data(self):
        file = pd.read_csv("/home/mathloverpi/currentfile/Note/Machine_learning/homework3/code/data.csv")
        df = pd.DataFrame(file)
        self.train_x = np.array(df["x"])
        self.train_y = np.array(df["y"])
        self.test_x = self.all_x
        self.test_y = self.all_y


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
        self.test_num =5 # samples.test_num
        self.line_x = np.arange(-4,4,8/20)
        # self.loss = samples.data
    def run(self):
        X_T = np.vstack((np.ones(len(self.train_x)),self.train_x))
        X = X_T.T
        Y = self.train_y.reshape(self.train_num,1)
        # print(X.T*X)
        self.w = np.dot(np.dot((np.linalg.inv(np.dot(X.T,X))),X.T),Y)
        self.y_train_pre = self.w[1]*self.train_x+self.w[0]
        self.y_test_pre = self.w[1]*self.test_x+self.w[0]
        self.line_y = self.w[1]*self.line_x+self.w[0]
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
        plt.plot(self.line_x, self.line_y,color='red',label='Predict Line') 
        plt.legend()
        plt.show()

class Ridge(object):
    def __init__(self, samples, Lambda=-6):
        """
       岭回归算法
        """
        self.Lambda = Lambda
        self.lambda_dt = 4/100
        self.line_x = np.arange(-4,4,8/20)
        self.train_x = samples.train_x
        self.train_y = samples.train_y
        self.test_x = samples.test_x
        self.test_y = samples.test_y
        self.all_w = []
        self.all_lambda=[]
        self.all_test_loss = []
        self.all_train_loss = []
        self.test_loss = []
        self.train_loss = []
        self.train_num = samples.train_num
        self.test_num = 5#samples.test_num
        # self.loss = samples.data
    def run(self):
        Lambda=self.Lambda
        for i in range(1):
            X_T = np.vstack((np.ones(len(self.train_x)),self.train_x))
            X = X_T.T
            Y = self.train_y.reshape(self.train_num,1)
            # print(X.T*X)
            self.w = np.dot(np.dot((np.linalg.inv(np.dot(X.T,X)+np.eye(np.shape(X)[1])*Lambda)),X.T),Y)
            self.y_train_pre = self.w[1]*self.train_x+self.w[0]
            self.y_test_pre = self.w[1]*self.test_x+self.w[0]
            self.line_y = self.w[1]*self.line_x+self.w[0]
            self.calculate_loss()
            print(self.w)
            print("Ridge Train Loss")
            print(self.train_loss)
            print("Ridge Test Loss")
            print(self.test_loss)
            self.plot_point()
            self.all_w.append(self.w)
            self.all_test_loss.append(self.test_loss)
            self.all_train_loss.append(self.train_loss)
            self.all_lambda.append(Lambda)
            print(i)
            Lambda=self.Lambda+i*self.lambda_dt
        self.plot_loss()

            # print(self.w)
        return 0
    def calculate_loss(self):
        self.test_loss =sum((self.y_test_pre - self.test_y) ** 2)/self.test_num
        self.train_loss = sum((self.y_train_pre - self.train_y) ** 2)/self.train_num
        return 0
    def plot_point(self):
        plt.scatter(self.train_x, self.train_y, alpha=0.6,label='Train Point')  # 绘制散点图,透明度为0.6（这样颜色浅一点,比较好看）
        plt.scatter(self.train_x, self.y_train_pre, alpha=0.6,label='Predict Train Point')  # 绘制散点图,透明度为0.6（这样颜色浅一点,比较好看）

        plt.scatter(self.test_x, self.test_y, alpha=0.6,label='Test Point')  # 绘制散点图,透明度为0.6（这样颜色浅一点,比较好看）
        plt.scatter(self.test_x, self.y_test_pre, alpha=0.6,label='Predict Test Point')  # 绘制散点图,透明度为0.6（这样颜色浅一点,比较好看）
        plt.plot(self.line_x, self.line_y,color='red',label='Predict Line') 

        plt.legend()
        plt.show()
        # plt.ion()
        # plt.pause(0.1)  #显示秒数
        # plt.close()
    def plot_loss(self):
        plt.plot(self.all_lambda, self.all_train_loss,color='red',label='Train Loss') 
        plt.plot(self.all_lambda, self.all_test_loss,color='blue',label='Test Loss') 
        plt.legend()
        plt.show()

    
def poly_test(x_train,x_test,y_train,y_test):
    rmses = []
    degrees = np.arange(1, 10)
    min_rmse, min_deg,score = 1e10, 0 ,0

    for deg in degrees:
        # 生成多项式特征集(如根据degree=3 ,生成 [[x,x**2,x**3]] )
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        x_train_poly = poly.fit_transform(x_train)

        # 多项式拟合
        poly_reg = LinearRegression()
        poly_reg.fit(x_train_poly, y_train)
        #print(poly_reg.coef_,poly_reg.intercept_) #系数及常数
        
        # 测试集比较
        x_test_poly = poly.fit_transform(x_test)
        y_test_pred = poly_reg.predict(x_test_poly)
        
        #mean_squared_error(y_true, y_pred) #均方误差回归损失,越小越好。
        poly_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        rmses.append(poly_rmse)
        # r2 范围[0，1]，R2越接近1拟合越好。
        r2score = r2_score(y_test, y_test_pred)
        
        # degree交叉验证
        if min_rmse > poly_rmse:
            min_rmse = poly_rmse
            min_deg = deg
            score = r2score
        print('degree = %s, RMSE = %.2f ,r2_score = %.2f' % (deg, poly_rmse,r2score))
            
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(degrees, rmses)
    ax.set_yscale('log')
    ax.set_xlabel('Degree')
    ax.set_ylabel('RMSE')
    ax.set_title('Best degree = %s, RMSE = %.2f, r2_score = %.2f' %(min_deg, min_rmse,score))  
    plt.show()    


if __name__ == "__main__":
    os.system('export DISPLAY=:0.0')
    samples = Samples(bound=[-2,2],point_num=5,train_num = 9)
    samples.plot_point()
    # LeastSquares = LeastSquares(samples)
    # LeastSquares.run()
    # Ridge = Ridge(samples)
    # Ridge.run()


    x_train = samples.train_x
    x_test  = samples.test_x
    y_train = samples.train_y
    y_test  = samples.test_y
    poly_test(x_train.reshape(-1, 1),x_test.reshape(-1, 1),y_train.reshape(-1, 1),y_test.reshape(-1, 1))

    # Huber = Huber(samples)
    # Huber.run()
