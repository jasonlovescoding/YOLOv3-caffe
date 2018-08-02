import caffe
import cv2

class UpsamplingBilinear2d(caffe.Layer):
    
    def setup(self, bottom, top):
        params = eval(self.param_str)
        zoom_factor = params['zoom_factor']
        
        # reshape is once-for-all
        n_out = int(bottom[0].num)
        self.w_out = int(bottom[0].width * zoom_factor)
        self.h_out = int(bottom[0].height * zoom_factor)
        c_out = int(bottom[0].channels)
        top[0].reshape(n_out, c_out, self.w_out, self.h_out)
            
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        for n in range(top[0].data[...].shape[0]):
            data = bottom[0].data[n, :, :, :]
            data = data.transpose((1, 2, 0))
            data = cv2.resize(data, (self.w_out, self.h_out), interpolation=cv2.INTER_NEAREST) 
            top[0].data[n, :, :, :] = data.transpose((2, 0, 1))

    def backward(self, top, propagate_down, bottom):
        pass