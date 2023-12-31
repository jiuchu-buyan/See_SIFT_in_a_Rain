import torch
import torch.nn as nn
import common

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

def make_model(args, parent=False):
    return EDSR(args)

class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        n_inch = args.n_inch
        n_outch = args.n_outch
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = common.MeanShift_48(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(n_inch, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.SENet(
                conv, 64, 64, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(4)
        ]

        m_body.append(conv(64, 128, kernel_size))
        for i in range(4):
            m_body.append(
                common.SENet(
                    conv, 128, 128, kernel_size, act=act, res_scale=args.res_scale
                ))

        m_body.append(conv(128, 256, kernel_size))

        for i in range(4):
            m_body.append(
                common.SENet(
                    conv, 256, 256, kernel_size, act=act, res_scale=args.res_scale
                ))

        m_body.append(conv(256, 128, kernel_size))

        for i in range(4):
            m_body.append(
                common.SENet(
                    conv, 128, 128, kernel_size, act=act, res_scale=args.res_scale
                ))

        m_body.append(conv(128, 64, kernel_size))

        for i in range(4):
            m_body.append(
                common.SENet(
                    conv, 64, 64, kernel_size, act=act, res_scale=args.res_scale
                ))

        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
           # common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_outch, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        #print(x.shape)

        x = self.sub_mean(x)
        #print(x.shape)

        hz_dc = x[:, 0:3, :, :]

        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        x += hz_dc

        x = self.add_mean(x)


        return x