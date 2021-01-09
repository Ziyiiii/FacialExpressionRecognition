import keras
from keras.layers import Dense, Dropout, Activation, Flatten,Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.utils import plot_model
batch_siz = 128
num_classes = 7
img_size=48
root_path='./fer2013'
model_path='./model'
class Model:
    def __init__(self):
        #声明一个成员变量model
        self.model = None

    def build_model(self):
        #建立序贯模型
        self.model = Sequential()

        self.model.add(Conv2D(32, (1, 1), strides=1, padding='same', input_shape=(img_size, img_size, 1)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (5, 5), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (5, 5), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(2048))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes))
        self.model.add(Activation('softmax'))
        self.model.summary()
    def train_model(self):
        sgd=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                optimizer=sgd,
                #optimizer='rmsprop',
                metrics=['accuracy'])
        #使用生成器来生成batch数据
        #首先进行图片预处理
        #自动扩充训练样本（旋转、翻转等）
        train_datagen = ImageDataGenerator(
        rescale = 1./255,   #重放缩因子，数值乘以1.0/255（归一化）
        shear_range = 0.2,  #剪切强度（逆时针方向的剪切变换角度）
        zoom_range = 0.2,   #随机缩放的幅度
            #进行随机水平翻转
        horizontal_flip=True)
        #归一化验证集
        val_datagen = ImageDataGenerator(
                rescale = 1./255)
        eval_datagen = ImageDataGenerator(
                rescale = 1./255)

        #flow_from_directory（dictory）方法
        #以文件分类名划分label
        train_generator = train_datagen.flow_from_directory(
                root_path+'/train', #dictory参数，该路径下的所有子文件夹的图片都会被生成器使用，无限产生batch数据
                target_size=(img_size,img_size), #图片将被resize成该尺寸
                color_mode='grayscale',     #颜色模式，graycsale或rgb（默认rgb）
                batch_size=batch_siz,       #batch数据的大小，默认为32
                class_mode='categorical')   #返回的标签形式，默认为‘category’，返回2D的独热码标签
        val_generator = val_datagen.flow_from_directory(
                root_path+'/val',   #同上
                target_size=(img_size,img_size),
                color_mode='grayscale',
                batch_size=batch_siz,
                class_mode='categorical')
        eval_generator = eval_datagen.flow_from_directory(
                root_path+'/test',  #同上
                target_size=(img_size,img_size),
                color_mode='grayscale',
                batch_size=batch_siz,
                class_mode='categorical')
        early_stopping = EarlyStopping(monitor='loss',patience=3)
        #利用生成器进行训练  调用fit_generator方法
        history_fit=self.model.fit_generator(
                train_generator,    #generator参数，生成训练集的生成器
                steps_per_epoch=800/(batch_siz/32),#28709 生成器返回这么多次数据时记为一个epoch结束，执行下一个epoch
                epochs=70,              #数据迭代轮数
                validation_data=val_generator,  #生成验证集的生成器
                validation_steps=2000,          #指定验证集的生成器返回次数
                #callbacks=[early_stopping]
                )

#       history_eval=self.model.evaluate_generator(
#               eval_generator,
#               steps=2000)
        #从一个生成器上来获取数据，进行预测
        history_predict=self.model.predict_generator(
                eval_generator,
                steps=2000)
        #保存训练日志
        with open(root_path+'/model_fit_log','w') as f:
            f.write(str(history_fit.history))
        with open(root_path+'/model_predict_log','w') as f:
            f.write(str(history_predict))
#         print("%s: %.2f%%" % (self.model.metrics_names[1], history_eval[1] * 100))
        print('model trained')
    def save_model(self):
        model_json=self.model.to_json()
        with open(model_path+"/model_json.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(model_path+'/model_weight.h5')
        self.model.save(model_path+'/model.h5')
        print('model saved')

if __name__=='__main__':
    model=Model()
    model.build_model()
    print('model built')
    model.train_model()
    print('model trained')
    model.save_model()
    print('model saved')
    # plot_model(model.model,to_file='model.png')
