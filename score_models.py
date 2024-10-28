import torch
import torch.nn.functional as F

from kans import KAN, FastKAN, ReLUKAN, BottleNeckKAGN, WavKAN, KABN, KACN, KAGN, KAJN, KALN
from kan_convs import KANConv2DLayer, FastKANConv2DLayer, ReLUKANConv2DLayer, BottleNeckKAGNConv2DLayer, WavKANConv2DLayer, KABNConv2DLayer, KACNConv2DLayer, KAGNConv2DLayer, KAJNConv2DLayer, KALNConv2DLayer, BottleNeckSelfKAGNtention2D
from functions import get_timestep_embedding

##----------------------------------------------------------------------------------------------------------##
##                                              SQuIRELS                                                    ##
##----------------------------------------------------------------------------------------------------------##
class MLP(torch.nn.Module):
    def __init__(self, input_dim, layer_widths, activate_final = False, activation_fn=F.relu):
        super(MLP, self).__init__()
        layers = []
        prev_width = input_dim
        for layer_width in layer_widths:
            layers.append(torch.nn.Linear(prev_width, layer_width))
            # # same init for everyone
            # torch.nn.init.constant_(layers[-1].weight, 0)
            prev_width = layer_width
        self.input_dim = input_dim
        self.layer_widths = layer_widths
        self.layers = torch.nn.ModuleList(layers)
        self.activate_final = activate_final
        self.activation_fn = activation_fn
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x)
        if self.activate_final:
            x = self.activation_fn(x)
        return x

class SquirelsScoreNetwork(torch.nn.Module):
    def __init__(self, encoder_layers=[256,256], pos_dim=128, decoder_layers=[256,256], x_dim=1, n_cond=1):

        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim *2
        self.locals = [encoder_layers, pos_dim, decoder_layers, x_dim]
        self.n_cond = n_cond


        self.net = MLP(3 * t_enc_dim + 1,
                       layer_widths=decoder_layers +[x_dim],
                       activate_final = False,
                       activation_fn=torch.nn.LeakyReLU())

        self.t_encoder = MLP(pos_dim,
                             layer_widths=encoder_layers +[t_enc_dim],
                             activate_final = False,
                             activation_fn=torch.nn.LeakyReLU())

        self.x_encoder = MLP(x_dim,
                             layer_widths=encoder_layers +[t_enc_dim],
                             activate_final = False,
                             activation_fn=torch.nn.LeakyReLU())
            
        self.e_encoder = MLP(1,
                             layer_widths=encoder_layers +[t_enc_dim],
                             activate_final = False,
                             activation_fn=torch.nn.LeakyReLU())
        
        
    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if len(selfcond.shape) == 1:
            selfcond = selfcond.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        
        xemb = self.x_encoder(x)
        
            
        if self.n_cond > 0:
            eemb = cond
            eemb = self.e_encoder(eemb)
            h = torch.cat([xemb, temb, selfcond, eemb], -1)
        else:
            h = torch.cat([xemb ,temb, selfcond], -1)
                
        out = self.net(h) 
        return out

class SquirelsScoreNetworkConv(torch.nn.Module):

    def __init__(self, encoder_layers=[256,256], temb_dim=128, conv_dof=32, x_dim=100, n_cond=0):
        super().__init__()
        self.conv_dof = conv_dof
        self.n_cond = n_cond
        
        self.temb_dim = temb_dim
        self.encoder_layers=encoder_layers
        self.t_enc_dim = self.conv_dof
        
        self.bias = False
        
        self.t_encoder = MLP(self.temb_dim,
                             layer_widths=self.encoder_layers +[self.t_enc_dim],
                             activate_final = False,
                             activation_fn=torch.nn.LeakyReLU())

        self.con_x_encoder1 = torch.nn.Conv2d(1, self.conv_dof, (3, 3), stride=(1, 1), 
                                              padding=(1, 1),bias=self.bias)
        self.con_x_encoder2 = torch.nn.Conv2d(self.conv_dof, self.conv_dof, (3, 3), stride=(1, 1), 
                                              padding=(1, 1),bias=self.bias)
        self.con_x_encoder3 = torch.nn.Conv2d(self.conv_dof, self.conv_dof, (3, 3), stride=(1, 1), 
                                              padding=(1, 1),bias=self.bias)
        self.con_x_encoder4 = torch.nn.Conv2d(self.conv_dof, self.conv_dof, (3, 3), stride=(1, 1), 
                                              padding=(1, 1),bias=self.bias)
        
        
        self.con_x_decoder1 = torch.nn.Conv2d(self.conv_dof*2+self.n_cond+1, self.conv_dof, (3, 3), stride=(1, 1), 
                                              padding=(1, 1),bias=self.bias)
        self.con_x_decoder2 = torch.nn.Conv2d(self.conv_dof, self.conv_dof, (3, 3), stride=(1, 1), 
                                              padding=(1, 1),bias=self.bias)
        self.con_x_decoder3 = torch.nn.Conv2d(self.conv_dof, self.conv_dof, (3, 3), stride=(1, 1), 
                                              padding=(1, 1),bias=self.bias)
        self.con_x_decoder4 = torch.nn.Conv2d(self.conv_dof, 1, (3, 3), stride=(1, 1), 
                                              padding=(1, 1),bias=self.bias)
        
        self.leakyReLU = torch.nn.LeakyReLU()

        
        
    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if self.n_cond > 0:
            cond = cond.view(-1, self.n_cond, 1, 1).expand(-1, -1, 10, 10)

        temb = get_timestep_embedding(t, self.temb_dim)
        #print(t.size(), temb.size())
        
        temb = self.t_encoder(temb).view(-1, self.t_enc_dim, 1, 1).expand(-1, -1, 10, 10)
        
        xemb = self.leakyReLU(self.con_x_encoder1(x))
        xemb = self.leakyReLU(self.con_x_encoder2(xemb))
        xemb = self.leakyReLU(self.con_x_encoder3(xemb))
        xemb = self.con_x_encoder4(xemb)
        
        if self.n_cond > 0:
            h = torch.cat([xemb, temb, selfcond, cond], 1)
        else:
            h = torch.cat([xemb, temb, selfcond], 1)
            
        out = self.leakyReLU(self.con_x_decoder1(h))
        out = self.leakyReLU(self.con_x_decoder2(out))
        out = self.leakyReLU(self.con_x_decoder3(out))
        out = self.con_x_decoder4(out)

        return out

class SquirelsScoreNetworkLinear(torch.nn.Module):

    def __init__(self, encoder_layers=[256,256], temb_dim=128, conv_dof=32, x_dim=100, n_cond=0):
        super().__init__()

        self.t_hl_dim=encoder_layers[0]
        self.t_nh_layers=len(encoder_layers)
        self.p_hl_dim=conv_dof
        self.p_nh_layers=4
        
        self.x_dim = x_dim
        self.temb_dim = temb_dim
        self.t_enc_layers = [self.temb_dim] + [self.t_hl_dim] * self.t_nh_layers + [self.p_hl_dim]
        self.p_enc_layers = [self.x_dim] + [self.p_hl_dim] * self.p_nh_layers
        self.p_dec_layers = [2 * self.p_hl_dim + self.x_dim + 3] + [self.p_hl_dim] * (self.p_nh_layers-1) + [self.x_dim]
        
        self.n_cond = n_cond      

        self.bias = False
        
        self.t_encoder = MLP(self.t_enc_layers[0],
                             layer_widths=self.t_enc_layers[1:],
                             activate_final = False,
                             activation_fn=torch.nn.LeakyReLU())
        
        self.con_x_encoder1 = MLP(self.p_enc_layers[0], [self.p_enc_layers[1]], activate_final = False)
        self.con_x_encoder2 = MLP(self.p_enc_layers[1], [self.p_enc_layers[2]], activate_final = False)
        self.con_x_encoder3 = MLP(self.p_enc_layers[2], [self.p_enc_layers[3]], activate_final = False)
        self.con_x_encoder4 = MLP(self.p_enc_layers[3], [self.p_enc_layers[4]], activate_final = False)

        self.con_x_decoder1 = MLP(self.p_dec_layers[0], [self.p_dec_layers[1]], activate_final = False)
        self.con_x_decoder2 = MLP(self.p_dec_layers[1], [self.p_dec_layers[2]], activate_final = False)
        self.con_x_decoder3 = MLP(self.p_dec_layers[2], [self.p_dec_layers[3]], activate_final = False)
        self.con_x_decoder4 = MLP(self.p_dec_layers[3], [self.p_dec_layers[4]], activate_final = False)
        
        self.leakyReLU = torch.nn.LeakyReLU()
                
        
    def forward(self, x, t, cond=None, selfcond=None):
        x_shape = x.shape
        x = x.reshape((x_shape[0], x_shape[1], x_shape[2]*x_shape[3]))

        if len(cond) > 0:
            cond = cond.unsqueeze(1)

        if len(selfcond) > 0:
            selfcond = selfcond.reshape((selfcond.shape[0], selfcond.shape[1], selfcond.shape[2]*selfcond.shape[3]))

        temb = get_timestep_embedding(t, self.temb_dim)
        
        temb = self.t_encoder(temb).unsqueeze(1)
        
        xemb = self.leakyReLU(self.con_x_encoder1(x))
        xemb = self.leakyReLU(self.con_x_encoder2(xemb))
        xemb = self.leakyReLU(self.con_x_encoder3(xemb))
        xemb = self.con_x_encoder4(xemb)
        
        # print(f"xemb: {xemb.shape}, temb: {temb.shape}, selfcond: {selfcond.shape}, cond: {cond.shape}")

        if self.n_cond > 0:
            h = torch.cat([xemb, temb, selfcond, cond], -1)
        else:
            h = torch.cat([xemb, temb, selfcond], -1)
            
        out = self.leakyReLU(self.con_x_decoder1(h))
        out = self.leakyReLU(self.con_x_decoder2(out))
        out = self.leakyReLU(self.con_x_decoder3(out))
        out = self.con_x_decoder4(out)

        out = out.reshape(x_shape)

        return out


##----------------------------------------------------------------------------------------------------------##
##                                                 KAN                                                      ##
##----------------------------------------------------------------------------------------------------------##

class ScoreKAN(torch.nn.Module):
    def __init__(self, encoder_layers=[256,256], pos_dim=128, decoder_layers=[256,256], x_dim=1, n_cond=1):

        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim *2
        self.locals = [encoder_layers, pos_dim, decoder_layers, x_dim]
        self.n_cond = n_cond


        self.net = KAN(layers_hidden = [3 * t_enc_dim + 1] + decoder_layers +[x_dim])
        self.t_encoder = KAN(layers_hidden = [pos_dim] + encoder_layers +[t_enc_dim])
        self.x_encoder = KAN(layers_hidden = [x_dim] + encoder_layers +[t_enc_dim])            
        self.e_encoder = KAN(layers_hidden = [1] + encoder_layers +[t_enc_dim])
        
        
    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if len(selfcond.shape) == 1:
            selfcond = selfcond.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        
        xemb = self.x_encoder(x)
        
            
        if self.n_cond > 0:
            eemb = cond
            eemb = self.e_encoder(eemb)
            h = torch.cat([xemb, temb, selfcond, eemb], -1)
        else:
            h = torch.cat([xemb ,temb, selfcond], -1)
                
        out = self.net(h) 
        return out

class ScoreKANConv(torch.nn.Module):

    def __init__(self, encoder_layers=[256,256], temb_dim=128, conv_dof=32, x_dim=100, n_cond=0):
        super().__init__()
        self.conv_dof = conv_dof
        self.n_cond = n_cond
        
        self.temb_dim = temb_dim
        self.encoder_layers = encoder_layers
        self.t_enc_dim = self.conv_dof
        
        self.bias = False
        
        self.t_encoder = KAN(layers_hidden = [self.temb_dim] + self.encoder_layers +[self.t_enc_dim])

        self.con_x_encoder1 = KANConv2DLayer(1, self.conv_dof, (3, 3), stride=1, padding=1)
        self.con_x_encoder2 = KANConv2DLayer(self.conv_dof, self.conv_dof, (3, 3), stride=1, padding=1)
        self.con_x_encoder3 = KANConv2DLayer(self.conv_dof, self.conv_dof, (3, 3), stride=1, padding=1)
        self.con_x_encoder4 = KANConv2DLayer(self.conv_dof, self.conv_dof, (3, 3), stride=1, padding=1)
        
        
        self.con_x_decoder1 = KANConv2DLayer(self.conv_dof*2+self.n_cond+1, self.conv_dof, (3, 3), stride=1, padding=1)
        self.con_x_decoder2 = KANConv2DLayer(self.conv_dof, self.conv_dof, (3, 3), stride=1, padding=1)
        self.con_x_decoder3 = KANConv2DLayer(self.conv_dof, self.conv_dof, (3, 3), stride=1, padding=1)
        self.con_x_decoder4 = KANConv2DLayer(self.conv_dof, 1, (3, 3), stride=(1, 1), padding=1)
        
        self.leakyReLU = torch.nn.LeakyReLU()

        
        
    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if self.n_cond > 0:
            cond = cond.view(-1, self.n_cond, 1, 1).expand(-1, -1, 10, 10)

        temb = get_timestep_embedding(t, self.temb_dim)
        #print(t.size(), temb.size())
        
        temb = self.t_encoder(temb).view(-1, self.t_enc_dim, 1, 1).expand(-1, -1, 10, 10)
        
        xemb = self.leakyReLU(self.con_x_encoder1(x))
        xemb = self.leakyReLU(self.con_x_encoder2(xemb))
        xemb = self.leakyReLU(self.con_x_encoder3(xemb))
        xemb = self.con_x_encoder4(xemb)
        
        if self.n_cond > 0:
            h = torch.cat([xemb, temb, selfcond, cond], 1)
        else:
            h = torch.cat([xemb, temb, selfcond], 1)
            
        out = self.leakyReLU(self.con_x_decoder1(h))
        out = self.leakyReLU(self.con_x_decoder2(out))
        out = self.leakyReLU(self.con_x_decoder3(out))
        out = self.con_x_decoder4(out)

        return out
    

##----------------------------------------------------------------------------------------------------------##
##                                                FAST KAN                                                  ##
##----------------------------------------------------------------------------------------------------------##

class FastScoreKAN(torch.nn.Module):
    def __init__(self, encoder_layers=[256,256], pos_dim=128, decoder_layers=[256,256], x_dim=1, n_cond=1):

        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim *2
        self.locals = [encoder_layers, pos_dim, decoder_layers, x_dim]
        self.n_cond = n_cond
        
        self.t_encoder = FastKAN(layers_hidden = [pos_dim] + encoder_layers + [t_enc_dim])
        self.x_encoder = FastKAN(layers_hidden = [x_dim] + encoder_layers + [t_enc_dim])            
        self.e_encoder = FastKAN(layers_hidden = [1] + encoder_layers + [t_enc_dim])
        self.net = FastKAN(layers_hidden = [3 * t_enc_dim + 1] + decoder_layers + [x_dim])
        
        
    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if len(selfcond.shape) == 1:
            selfcond = selfcond.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        
        xemb = self.x_encoder(x)
        
            
        if self.n_cond > 0:
            eemb = cond
            eemb = self.e_encoder(eemb)
            h = torch.cat([xemb, temb, selfcond, eemb], -1)
        else:
            h = torch.cat([xemb ,temb, selfcond], -1)
                
        out = self.net(h) 
        return out

class FastScoreKANConv(torch.nn.Module):

    def __init__(self, encoder_layers=[256,256], temb_dim=128, conv_dof=32, x_dim=100, n_cond=0):
        super().__init__()
        self.conv_dof = conv_dof
        self.n_cond = n_cond
        
        self.temb_dim = temb_dim
        self.encoder_layers = encoder_layers
        self.t_enc_dim = self.conv_dof
        
        self.bias = False
        
        self.t_encoder = FastKAN(layers_hidden = [self.temb_dim] + self.encoder_layers +[self.t_enc_dim])

        self.con_x_encoder1 = FastKANConv2DLayer(1, self.conv_dof, (3, 3), stride=1, padding=1)
        self.con_x_encoder2 = FastKANConv2DLayer(self.conv_dof, self.conv_dof, (3, 3), stride=1, padding=1)
        self.con_x_encoder3 = FastKANConv2DLayer(self.conv_dof, self.conv_dof, (3, 3), stride=1, padding=1)
        self.con_x_encoder4 = FastKANConv2DLayer(self.conv_dof, self.conv_dof, (3, 3), stride=1, padding=1)
        
        
        self.con_x_decoder1 = FastKANConv2DLayer(self.conv_dof*2+self.n_cond+1, self.conv_dof, (3, 3), stride=1, padding=1)
        self.con_x_decoder2 = FastKANConv2DLayer(self.conv_dof, self.conv_dof, (3, 3), stride=1, padding=1)
        self.con_x_decoder3 = FastKANConv2DLayer(self.conv_dof, self.conv_dof, (3, 3), stride=1, padding=1)
        self.con_x_decoder4 = FastKANConv2DLayer(self.conv_dof, 1, (3, 3), stride=(1, 1), padding=1)
                
        
    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if self.n_cond > 0:
            cond = cond.view(-1, self.n_cond, 1, 1).expand(-1, -1, 10, 10)

        temb = get_timestep_embedding(t, self.temb_dim)
        #print(t.size(), temb.size())
        
        temb = self.t_encoder(temb).view(-1, self.t_enc_dim, 1, 1).expand(-1, -1, 10, 10)
        
        xemb = self.con_x_encoder1(x)
        xemb = self.con_x_encoder2(xemb)
        xemb = self.con_x_encoder3(xemb)
        xemb = self.con_x_encoder4(xemb)
        
        if self.n_cond > 0:
            h = torch.cat([xemb, temb, selfcond, cond], 1)
        else:
            h = torch.cat([xemb, temb, selfcond], 1)
            
        out = self.con_x_decoder1(h)
        out = self.con_x_decoder2(out)
        out = self.con_x_decoder3(out)
        out = self.con_x_decoder4(out)

        return out
    
class FastScoreKANLinear(torch.nn.Module):

    def __init__(self, encoder_layers=[256,256], temb_dim=128, conv_dof=32, x_dim=100, n_cond=0):
        super().__init__()

        self.t_hl_dim=encoder_layers[0]
        self.t_nh_layers=len(encoder_layers)
        self.p_hl_dim=conv_dof
        self.p_nh_layers=4
        
        self.x_dim = x_dim
        self.temb_dim = temb_dim
        self.t_enc_layers = [self.temb_dim] + [self.t_hl_dim] * self.t_nh_layers + [self.p_hl_dim]
        self.p_enc_layers = [self.x_dim] + [self.p_hl_dim] * self.p_nh_layers
        self.p_dec_layers = [2 * self.p_hl_dim + self.x_dim + 3] + [self.p_hl_dim] * (self.p_nh_layers-1) + [self.x_dim]
        
        self.n_cond = n_cond      

        self.bias = False
        
        self.t_encoder = FastKAN(layers_hidden = self.t_enc_layers)

        self.con_x_encoder = FastKAN(layers_hidden = self.p_enc_layers)
        
        self.con_x_decoder = FastKAN(layers_hidden = self.p_dec_layers)
                
        
    def forward(self, x, t, cond=None, selfcond=None):
        x_shape = x.shape
        x = x.reshape((x_shape[0], x_shape[1], x_shape[2]*x_shape[3]))

        if len(cond) > 0:
            cond = cond.unsqueeze(1)

        if len(selfcond) > 0:
            selfcond = selfcond.reshape((selfcond.shape[0], selfcond.shape[1], selfcond.shape[2]*selfcond.shape[3]))

        temb = get_timestep_embedding(t, self.temb_dim)
        
        temb = self.t_encoder(temb).unsqueeze(1)
        
        xemb = self.con_x_encoder(x)
        
        # print(f"xemb: {xemb.shape}, temb: {temb.shape}, selfcond: {selfcond.shape}, cond: {cond.shape}")

        if self.n_cond > 0:
            h = torch.cat([xemb, temb, selfcond, cond], -1)
        else:
            h = torch.cat([xemb, temb, selfcond], -1)
            
        out = self.con_x_decoder(h)

        out = out.reshape(x_shape)

        return out
    

##----------------------------------------------------------------------------------------------------------##
##                                              FAST KAN WIDE                                               ##
##----------------------------------------------------------------------------------------------------------##

class FastScoreKANWide(torch.nn.Module):
    def __init__(self, encoder_layers=[256,256], pos_dim=128, decoder_layers=[256,256], x_dim=1, n_cond=1):

        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim *2
        self.locals = [encoder_layers, pos_dim, decoder_layers, x_dim]
        self.n_cond = n_cond


        self.net = FastKAN(layers_hidden = [3 * t_enc_dim + 1] + decoder_layers +[x_dim])
        self.t_encoder = FastKAN(layers_hidden = [pos_dim] + encoder_layers +[t_enc_dim])
        self.x_encoder = FastKAN(layers_hidden = [x_dim] + encoder_layers +[t_enc_dim])            
        self.e_encoder = FastKAN(layers_hidden = [1] + encoder_layers +[t_enc_dim])
        
        
    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if len(selfcond.shape) == 1:
            selfcond = selfcond.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        
        xemb = self.x_encoder(x)
        
            
        if self.n_cond > 0:
            eemb = cond
            eemb = self.e_encoder(eemb)
            h = torch.cat([xemb, temb, selfcond, eemb], -1)
        else:
            h = torch.cat([xemb ,temb, selfcond], -1)
                
        out = self.net(h) 
        return out

class FastScoreKANConvWide(torch.nn.Module):

    def __init__(self, encoder_layers=[256,256], temb_dim=128, conv_dof=32, x_dim=100, n_cond=0):
        super().__init__()
        self.conv_dof = conv_dof
        self.n_cond = n_cond
        
        self.temb_dim = temb_dim
        self.encoder_layers = encoder_layers
        self.t_enc_dim = self.conv_dof
        
        self.bias = False
        
        self.t_encoder = FastKAN(layers_hidden = [self.temb_dim] + self.encoder_layers +[self.t_enc_dim])

        self.con_x_encoder = FastKANConv2DLayer(1, self.conv_dof, (3, 3), stride=1, padding=1)
        # self.con_x_encoder1 = FastKANConv2DLayer(1, self.conv_dof, (3, 3), stride=1, padding=1)
        # self.con_x_encoder2 = FastKANConv2DLayer(self.conv_dof, self.conv_dof, (3, 3), stride=1, padding=1)
        # self.con_x_encoder3 = FastKANConv2DLayer(self.conv_dof, self.conv_dof, (3, 3), stride=1, padding=1)
        # self.con_x_encoder4 = FastKANConv2DLayer(self.conv_dof, self.conv_dof, (3, 3), stride=1, padding=1)
        
        self.con_x_decoder = FastKANConv2DLayer(self.conv_dof*2+self.n_cond+1, 1, (3, 3), stride=1, padding=1)
        # self.con_x_decoder1 = FastKANConv2DLayer(self.conv_dof*2+self.n_cond+1, self.conv_dof, (3, 3), stride=1, padding=1)
        # self.con_x_decoder2 = FastKANConv2DLayer(self.conv_dof, self.conv_dof, (3, 3), stride=1, padding=1)
        # self.con_x_decoder3 = FastKANConv2DLayer(self.conv_dof, self.conv_dof, (3, 3), stride=1, padding=1)
        # self.con_x_decoder4 = FastKANConv2DLayer(self.conv_dof, 1, (3, 3), stride=(1, 1), padding=1)
                
        
    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if self.n_cond > 0:
            cond = cond.view(-1, self.n_cond, 1, 1).expand(-1, -1, 10, 10)

        temb = get_timestep_embedding(t, self.temb_dim)
        #print(t.size(), temb.size())
        
        temb = self.t_encoder(temb).view(-1, self.t_enc_dim, 1, 1).expand(-1, -1, 10, 10)
        
        xemb = self.con_x_encoder(x)
        # xemb = self.con_x_encoder1(x)
        # xemb = self.con_x_encoder2(xemb)
        # xemb = self.con_x_encoder3(xemb)
        # xemb = self.con_x_encoder4(xemb)
        
        if self.n_cond > 0:
            h = torch.cat([xemb, temb, selfcond, cond], 1)
        else:
            h = torch.cat([xemb, temb, selfcond], 1)

        out = self.con_x_decoder(h)
        # out = self.con_x_decoder1(h)
        # out = self.con_x_decoder2(out)
        # out = self.con_x_decoder3(out)
        # out = self.con_x_decoder4(out)

        return out
    

#----------------------------------------------------------------------------------------------------------#
#                                                 RELU KAN                                                 #
#----------------------------------------------------------------------------------------------------------#

class ReluScoreKAN(torch.nn.Module):
    def __init__(self, encoder_layers=[256,256], pos_dim=128, decoder_layers=[256,256], x_dim=1, n_cond=1):

        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim *2
        self.locals = [encoder_layers, pos_dim, decoder_layers, x_dim]
        self.n_cond = n_cond


        self.net = ReLUKAN(layers_hidden = [3 * t_enc_dim + 1] + decoder_layers +[x_dim])
        self.t_encoder = ReLUKAN(layers_hidden = [pos_dim] + encoder_layers +[t_enc_dim])
        self.x_encoder = ReLUKAN(layers_hidden = [x_dim] + encoder_layers +[t_enc_dim])            
        self.e_encoder = ReLUKAN(layers_hidden = [1] + encoder_layers +[t_enc_dim])
        
        
    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if len(selfcond.shape) == 1:
            selfcond = selfcond.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        
        xemb = self.x_encoder(x)
        
            
        if self.n_cond > 0:
            eemb = cond
            eemb = self.e_encoder(eemb)
            h = torch.cat([xemb, temb, selfcond, eemb], -1)
        else:
            h = torch.cat([xemb ,temb, selfcond], -1)
                
        out = self.net(h) 
        return out

class ReluScoreKANConv(torch.nn.Module):

    def __init__(self, encoder_layers=[256,256], temb_dim=128, conv_dof=32, x_dim=100, n_cond=0):
        super().__init__()
        self.conv_dof = conv_dof
        self.n_cond = n_cond
        
        self.temb_dim = temb_dim
        self.encoder_layers = encoder_layers
        self.t_enc_dim = self.conv_dof
        
        self.bias = False
        
        self.t_encoder = ReLUKAN(layers_hidden = [self.temb_dim] + self.encoder_layers +[self.t_enc_dim])

        self.con_x_encoder1 = ReLUKANConv2DLayer(1, self.conv_dof, (3, 3), stride=1, padding=1)
        self.con_x_encoder2 = ReLUKANConv2DLayer(self.conv_dof, self.conv_dof, (3, 3), stride=1, padding=1)
        self.con_x_encoder3 = ReLUKANConv2DLayer(self.conv_dof, self.conv_dof, (3, 3), stride=1, padding=1)
        self.con_x_encoder4 = ReLUKANConv2DLayer(self.conv_dof, self.conv_dof, (3, 3), stride=1, padding=1)
        
        
        self.con_x_decoder1 = ReLUKANConv2DLayer(self.conv_dof*2+self.n_cond+1, self.conv_dof, (3, 3), stride=1, padding=1)
        self.con_x_decoder2 = ReLUKANConv2DLayer(self.conv_dof, self.conv_dof, (3, 3), stride=1, padding=1)
        self.con_x_decoder3 = ReLUKANConv2DLayer(self.conv_dof, self.conv_dof, (3, 3), stride=1, padding=1)
        self.con_x_decoder4 = ReLUKANConv2DLayer(self.conv_dof, 1, (3, 3), stride=(1, 1), padding=1)


    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if self.n_cond > 0:
            cond = cond.view(-1, self.n_cond, 1, 1).expand(-1, -1, 10, 10)

        temb = get_timestep_embedding(t, self.temb_dim)
        #print(t.size(), temb.size())
        
        temb = self.t_encoder(temb).view(-1, self.t_enc_dim, 1, 1).expand(-1, -1, 10, 10)
        
        xemb = self.con_x_encoder1(x)
        xemb = self.con_x_encoder2(xemb)
        xemb = self.con_x_encoder3(xemb)
        xemb = self.con_x_encoder4(xemb)
        
        if self.n_cond > 0:
            h = torch.cat([xemb, temb, selfcond, cond], 1)
        else:
            h = torch.cat([xemb, temb, selfcond], 1)
            
        out = self.con_x_decoder1(h)
        out = self.con_x_decoder2(out)
        out = self.con_x_decoder3(out)
        out = self.con_x_decoder4(out)

        return out

class ReluScoreKANLinear(torch.nn.Module):

    def __init__(self, encoder_layers=[256,256], temb_dim=128, conv_dof=32, x_dim=100, n_cond=0):
        super().__init__()

        self.t_hl_dim=encoder_layers[0]
        self.t_nh_layers=len(encoder_layers)
        self.p_hl_dim=conv_dof
        self.p_nh_layers=4
        
        self.x_dim = x_dim
        self.temb_dim = temb_dim
        self.t_enc_layers = [self.temb_dim] + [self.t_hl_dim] * self.t_nh_layers + [self.p_hl_dim]
        self.p_enc_layers = [self.x_dim] + [self.p_hl_dim] * self.p_nh_layers
        self.p_dec_layers = [2 * self.p_hl_dim + self.x_dim + 3] + [self.p_hl_dim] * (self.p_nh_layers-1) + [self.x_dim]
        
        self.n_cond = n_cond      

        self.bias = False
        
        self.t_encoder = ReLUKAN(layers_hidden = self.t_enc_layers)

        self.con_x_encoder = ReLUKAN(layers_hidden = self.p_enc_layers)
        
        self.con_x_decoder = ReLUKAN(layers_hidden = self.p_dec_layers)
                
        
    def forward(self, x, t, cond=None, selfcond=None):
        x_shape = x.shape
        x = x.reshape((x_shape[0], x_shape[1], x_shape[2]*x_shape[3]))

        if len(cond) > 0:
            cond = cond.unsqueeze(1)

        if len(selfcond) > 0:
            selfcond = selfcond.reshape((selfcond.shape[0], selfcond.shape[1], selfcond.shape[2]*selfcond.shape[3]))

        temb = get_timestep_embedding(t, self.temb_dim)
        
        temb = self.t_encoder(temb).unsqueeze(1)
        
        xemb = self.con_x_encoder(x).unsqueeze(1)

        if self.n_cond > 0:
            h = torch.cat([xemb, temb, selfcond, cond], -1)
        else:
            h = torch.cat([xemb, temb, selfcond], -1)
            
        out = self.con_x_decoder(h)

        out = out.reshape(x_shape)

        return out

#----------------------------------------------------------------------------------------------------------#
#                                              BOTTLENECK KAN                                              #
#----------------------------------------------------------------------------------------------------------#

class BottleneckScoreKAGN(torch.nn.Module):
    def __init__(self, encoder_layers=[256,256], pos_dim=128, decoder_layers=[256,256], x_dim=1, n_cond=1):

        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim *2
        self.locals = [encoder_layers, pos_dim, decoder_layers, x_dim]
        self.n_cond = n_cond


        self.net = BottleNeckKAGN(layers_hidden = [3 * t_enc_dim + 1] + decoder_layers +[x_dim])
        self.t_encoder = BottleNeckKAGN(layers_hidden = [pos_dim] + encoder_layers +[t_enc_dim])
        self.x_encoder = BottleNeckKAGN(layers_hidden = [x_dim] + encoder_layers +[t_enc_dim])            
        self.e_encoder = BottleNeckKAGN(layers_hidden = [1] + encoder_layers +[t_enc_dim])
        
        
    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if len(selfcond.shape) == 1:
            selfcond = selfcond.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)

        xemb = self.x_encoder(x)
        
            
        if self.n_cond > 0:
            eemb = cond
            eemb = self.e_encoder(eemb)
            h = torch.cat([xemb, temb, selfcond, eemb], -1)
        else:
            h = torch.cat([xemb ,temb, selfcond], -1)
                
        out = self.net(h) 
        return out

class BottleneckScoreKAGNConv(torch.nn.Module):

    def __init__(self, encoder_layers=[256,256], temb_dim=128, conv_dof=32, x_dim=100, n_cond=0):
        super().__init__()
        self.conv_dof = conv_dof
        self.n_cond = n_cond
        
        self.temb_dim = temb_dim
        self.encoder_layers = encoder_layers
        self.t_enc_dim = self.conv_dof
        
        self.bias = False
        
        self.t_encoder = BottleNeckKAGN(layers_hidden = [self.temb_dim] + self.encoder_layers +[self.t_enc_dim])

        self.con_x_encoder1 = BottleNeckKAGNConv2DLayer(1, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_encoder2 = BottleNeckKAGNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_encoder3 = BottleNeckKAGNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_encoder4 = BottleNeckKAGNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        
        
        self.con_x_decoder1 = BottleNeckKAGNConv2DLayer(self.conv_dof*2+self.n_cond+1, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_decoder2 = BottleNeckKAGNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_decoder3 = BottleNeckKAGNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_decoder4 = BottleNeckKAGNConv2DLayer(self.conv_dof, 1, 3, stride=(1, 1), padding=1)

        
    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if self.n_cond > 0:
            cond = cond.view(-1, self.n_cond, 1, 1).expand(-1, -1, 10, 10)

        temb = get_timestep_embedding(t, self.temb_dim)
        #print(t.size(), temb.size())
        
        temb = self.t_encoder(temb).view(-1, self.t_enc_dim, 1, 1).expand(-1, -1, 10, 10)
        
        xemb = self.con_x_encoder1(x)
        xemb = self.con_x_encoder2(xemb)
        xemb = self.con_x_encoder3(xemb)
        xemb = self.con_x_encoder4(xemb)
        
        if self.n_cond > 0:
            h = torch.cat([xemb, temb, selfcond, cond], 1)
        else:
            h = torch.cat([xemb, temb, selfcond], 1)
            
        out = self.con_x_decoder1(h)
        out = self.con_x_decoder2(out)
        out = self.con_x_decoder3(out)
        out = self.con_x_decoder4(out)

        return out

class BottleneckScoreKAGNLinear(torch.nn.Module):

    def __init__(self, encoder_layers=[256,256], temb_dim=128, conv_dof=32, x_dim=100, n_cond=0):
        super().__init__()

        self.t_hl_dim=encoder_layers[0]
        self.t_nh_layers=len(encoder_layers)
        self.p_hl_dim=conv_dof
        self.p_nh_layers=4
        
        self.x_dim = x_dim
        self.temb_dim = temb_dim
        self.t_enc_layers = [self.temb_dim] + [self.t_hl_dim] * self.t_nh_layers + [self.p_hl_dim]
        self.p_enc_layers = [self.x_dim] + [self.p_hl_dim] * self.p_nh_layers
        self.p_dec_layers = [2 * self.p_hl_dim + self.x_dim + 3] + [self.p_hl_dim] * (self.p_nh_layers-1) + [self.x_dim]
        
        self.n_cond = n_cond      

        self.bias = False
        
        self.t_encoder = BottleNeckKAGN(layers_hidden = self.t_enc_layers)

        self.con_x_encoder = BottleNeckKAGN(layers_hidden = self.p_enc_layers)
        
        self.con_x_decoder = BottleNeckKAGN(layers_hidden = self.p_dec_layers)
                
        
    def forward(self, x, t, cond=None, selfcond=None):
        x_shape = x.shape
        x = x.reshape((x_shape[0], x_shape[1], x_shape[2]*x_shape[3]))
        x = torch.squeeze(x)

        # if len(cond) > 0:
        #     cond = cond.unsqueeze(1)

        if len(selfcond) > 0:
            selfcond = selfcond.reshape((selfcond.shape[0], selfcond.shape[2]*selfcond.shape[3]))

        temb = get_timestep_embedding(t, self.temb_dim)
        
        temb = self.t_encoder(temb).unsqueeze(1)
        temb = torch.squeeze(temb)
        
        xemb = self.con_x_encoder(x)
        
        # print(f"xemb: {xemb.shape}, temb: {temb.shape}, selfcond: {selfcond.shape}, cond: {cond.shape}")

        if self.n_cond > 0:
            h = torch.cat([xemb, temb, selfcond, cond], -1)
        else:
            h = torch.cat([xemb, temb, selfcond], -1)
            
        out = self.con_x_decoder(h)

        out = out.reshape(x_shape)

        return out

class BottleneckScoreKAGNAttentionConv(torch.nn.Module):

    def __init__(self, encoder_layers=[256,256], temb_dim=128, conv_dof=32, x_dim=100, n_cond=0):
        super().__init__()
        self.conv_dof = conv_dof
        self.n_cond = n_cond
        
        self.temb_dim = temb_dim
        self.encoder_layers = encoder_layers
        self.t_enc_dim = self.conv_dof
        
        self.bias = False
        
        self.t_encoder = BottleNeckKAGN(layers_hidden = [self.temb_dim] + self.encoder_layers +[self.t_enc_dim])

        self.con_x_encoder1 = BottleNeckKAGNConv2DLayer(1, self.conv_dof, kernel_size=1, stride=1, padding=0)
        self.con_x_encoder2 = BottleNeckSelfKAGNtention2D(self.conv_dof, kernel_size=3, stride=1, padding=1)
        self.con_x_encoder3 = BottleNeckSelfKAGNtention2D(self.conv_dof, kernel_size=3, stride=1, padding=1)
        # self.con_x_encoder4 = BottleNeckSelfKAGNtention2D(self.conv_dof, kernel_size=3, stride=1, padding=1)
                
        self.con_x_decoder0 = BottleNeckKAGNConv2DLayer(self.conv_dof*2+self.n_cond+1, self.conv_dof, 1, stride=1, padding=0)
        
        # self.con_x_decoder1 = BottleNeckSelfKAGNtention2D(self.conv_dof, kernel_size=3, stride=1, padding=1)
        self.con_x_decoder2 = BottleNeckSelfKAGNtention2D(self.conv_dof, kernel_size=3, stride=1, padding=1)
        self.con_x_decoder3 = BottleNeckSelfKAGNtention2D(self.conv_dof, kernel_size=3, stride=1, padding=1)
        self.con_x_decoder4 = BottleNeckKAGNConv2DLayer(self.conv_dof, 1, 1, stride=1, padding=0)

        
    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if self.n_cond > 0:
            cond = cond.view(-1, self.n_cond, 1, 1).expand(-1, -1, 10, 10)

        temb = get_timestep_embedding(t, self.temb_dim)
        #print(t.size(), temb.size())
        
        temb = self.t_encoder(temb).view(-1, self.t_enc_dim, 1, 1).expand(-1, -1, 10, 10)
        
        xemb = self.con_x_encoder1(x)
        xemb = self.con_x_encoder2(xemb)
        xemb = self.con_x_encoder3(xemb)
        # xemb = self.con_x_encoder4(xemb)
        
        if self.n_cond > 0:
            h = torch.cat([xemb, temb, selfcond, cond], 1)
        else:
            h = torch.cat([xemb, temb, selfcond], 1)
        
        out = self.con_x_decoder0(h)
        # out = self.con_x_decoder1(out)
        out = self.con_x_decoder2(out)
        out = self.con_x_decoder3(out)
        out = self.con_x_decoder4(out)

        return out
    


#----------------------------------------------------------------------------------------------------------#
#                                                WAVELET KAN                                               #
#----------------------------------------------------------------------------------------------------------#

class WavScoreKAN(torch.nn.Module):
    def __init__(self, encoder_layers=[256,256], pos_dim=128, decoder_layers=[256,256], x_dim=1, n_cond=1):

        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim *2
        self.locals = [encoder_layers, pos_dim, decoder_layers, x_dim]
        self.n_cond = n_cond


        self.net = WavKAN(layers_hidden = [3 * t_enc_dim + 1] + decoder_layers +[x_dim])
        self.t_encoder = WavKAN(layers_hidden = [pos_dim] + encoder_layers +[t_enc_dim])
        self.x_encoder = WavKAN(layers_hidden = [x_dim] + encoder_layers +[t_enc_dim])            
        self.e_encoder = WavKAN(layers_hidden = [1] + encoder_layers +[t_enc_dim])
        
        
    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if len(selfcond.shape) == 1:
            selfcond = selfcond.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        
        xemb = self.x_encoder(x)
        
            
        if self.n_cond > 0:
            eemb = cond
            eemb = self.e_encoder(eemb)
            h = torch.cat([xemb, temb, selfcond, eemb], -1)
        else:
            h = torch.cat([xemb ,temb, selfcond], -1)
                
        out = self.net(h) 
        return out

class WavScoreKANConv(torch.nn.Module):

    def __init__(self, encoder_layers=[256,256], temb_dim=128, conv_dof=32, x_dim=100, n_cond=0):
        super().__init__()
        self.conv_dof = conv_dof
        self.n_cond = n_cond
        
        self.temb_dim = temb_dim
        self.encoder_layers = encoder_layers
        self.t_enc_dim = self.conv_dof
        
        self.bias = False
        
        self.t_encoder = WavKAN(layers_hidden = [self.temb_dim] + self.encoder_layers +[self.t_enc_dim])

        self.con_x_encoder1 = WavKANConv2DLayer(1, self.conv_dof, (3, 3), stride=1, padding=1)
        self.con_x_encoder2 = WavKANConv2DLayer(self.conv_dof, self.conv_dof, (3, 3), stride=1, padding=1)
        self.con_x_encoder3 = WavKANConv2DLayer(self.conv_dof, self.conv_dof, (3, 3), stride=1, padding=1)
        self.con_x_encoder4 = WavKANConv2DLayer(self.conv_dof, self.conv_dof, (3, 3), stride=1, padding=1)
        
        
        self.con_x_decoder1 = WavKANConv2DLayer(self.conv_dof*2+self.n_cond+1, self.conv_dof, (3, 3), stride=1, padding=1)
        self.con_x_decoder2 = WavKANConv2DLayer(self.conv_dof, self.conv_dof, (3, 3), stride=1, padding=1)
        self.con_x_decoder3 = WavKANConv2DLayer(self.conv_dof, self.conv_dof, (3, 3), stride=1, padding=1)
        self.con_x_decoder4 = WavKANConv2DLayer(self.conv_dof, 1, (3, 3), stride=(1, 1), padding=1)
       

    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if self.n_cond > 0:
            cond = cond.view(-1, self.n_cond, 1, 1).expand(-1, -1, 10, 10)

        temb = get_timestep_embedding(t, self.temb_dim)
        #print(t.size(), temb.size())
        
        temb = self.t_encoder(temb).view(-1, self.t_enc_dim, 1, 1).expand(-1, -1, 10, 10)
        
        xemb = self.con_x_encoder1(x)
        xemb = self.con_x_encoder2(xemb)
        xemb = self.con_x_encoder3(xemb)
        xemb = self.con_x_encoder4(xemb)
        
        if self.n_cond > 0:
            h = torch.cat([xemb, temb, selfcond, cond], 1)
        else:
            h = torch.cat([xemb, temb, selfcond], 1)
            
        out = self.con_x_decoder1(h)
        out = self.con_x_decoder2(out)
        out = self.con_x_decoder3(out)
        out = self.con_x_decoder4(out)

        return out

class WavScoreKANLinear(torch.nn.Module):

    def __init__(self, encoder_layers=[256,256], temb_dim=128, conv_dof=32, x_dim=100, n_cond=0):
        super().__init__()

        self.t_hl_dim=encoder_layers[0]
        self.t_nh_layers=len(encoder_layers)
        self.p_hl_dim=conv_dof
        self.p_nh_layers=4
        
        self.x_dim = x_dim
        self.temb_dim = temb_dim
        self.t_enc_layers = [self.temb_dim] + [self.t_hl_dim] * self.t_nh_layers + [self.p_hl_dim]
        self.p_enc_layers = [self.x_dim] + [self.p_hl_dim] * self.p_nh_layers
        self.p_dec_layers = [2 * self.p_hl_dim + self.x_dim + 3] + [self.p_hl_dim] * (self.p_nh_layers-1) + [self.x_dim]
        
        self.n_cond = n_cond      

        self.bias = False
        
        self.t_encoder = WavKAN(layers_hidden = self.t_enc_layers)

        self.con_x_encoder = WavKAN(layers_hidden = self.p_enc_layers)
        
        self.con_x_decoder = WavKAN(layers_hidden = self.p_dec_layers)
                
        
    def forward(self, x, t, cond=None, selfcond=None):
        x_shape = x.shape
        x = x.reshape((x_shape[0], x_shape[1], x_shape[2]*x_shape[3]))

        if len(cond) > 0:
            cond = cond.unsqueeze(1)

        if len(selfcond) > 0:
            selfcond = selfcond.reshape((selfcond.shape[0], selfcond.shape[1], selfcond.shape[2]*selfcond.shape[3]))

        temb = get_timestep_embedding(t, self.temb_dim)
        
        temb = self.t_encoder(temb).unsqueeze(1)
        
        xemb = self.con_x_encoder(x)
        
        # print(f"xemb: {xemb.shape}, temb: {temb.shape}, selfcond: {selfcond.shape}, cond: {cond.shape}")

        if self.n_cond > 0:
            h = torch.cat([xemb, temb, selfcond, cond], -1)
        else:
            h = torch.cat([xemb, temb, selfcond], -1)
            
        out = self.con_x_decoder(h)

        out = out.reshape(x_shape)

        return out
    
#----------------------------------------------------------------------------------------------------------#
#                                               BERNSTEIN KAN                                              #
#----------------------------------------------------------------------------------------------------------#
# TypeError: randn(): argument 'size' must be tuple of ints, but found element of type tuple at pos 4

class BernScoreKAN(torch.nn.Module):
    def __init__(self, encoder_layers=[256,256], pos_dim=128, decoder_layers=[256,256], x_dim=1, n_cond=1):

        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim *2
        self.locals = [encoder_layers, pos_dim, decoder_layers, x_dim]
        self.n_cond = n_cond


        self.net = KABN(layers_hidden = [3 * t_enc_dim + 1] + decoder_layers +[x_dim])
        self.t_encoder = KABN(layers_hidden = [pos_dim] + encoder_layers +[t_enc_dim])
        self.x_encoder = KABN(layers_hidden = [x_dim] + encoder_layers +[t_enc_dim])            
        self.e_encoder = KABN(layers_hidden = [1] + encoder_layers +[t_enc_dim])
        
        
    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if len(selfcond.shape) == 1:
            selfcond = selfcond.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        
        xemb = self.x_encoder(x)
        
            
        if self.n_cond > 0:
            eemb = cond
            eemb = self.e_encoder(eemb)
            h = torch.cat([xemb, temb, selfcond, eemb], -1)
        else:
            h = torch.cat([xemb ,temb, selfcond], -1)
                
        out = self.net(h) 
        return out

class BernScoreKANConv(torch.nn.Module):

    def __init__(self, encoder_layers=[256,256], temb_dim=128, conv_dof=32, x_dim=100, n_cond=0):
        super().__init__()
        self.conv_dof = conv_dof
        self.n_cond = n_cond
        
        self.temb_dim = temb_dim
        self.encoder_layers = encoder_layers
        self.t_enc_dim = self.conv_dof
        
        self.bias = False
        
        self.t_encoder = KABN(layers_hidden = [self.temb_dim] + self.encoder_layers +[self.t_enc_dim])

        self.con_x_encoder1 = KABNConv2DLayer(1, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_encoder2 = KABNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_encoder3 = KABNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_encoder4 = KABNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        
        
        self.con_x_decoder1 = KABNConv2DLayer(self.conv_dof*2+self.n_cond+1, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_decoder2 = KABNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_decoder3 = KABNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_decoder4 = KABNConv2DLayer(self.conv_dof, 1, 3, stride=1, padding=1)
        

    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if self.n_cond > 0:
            cond = cond.view(-1, self.n_cond, 1, 1).expand(-1, -1, 10, 10)

        temb = get_timestep_embedding(t, self.temb_dim)
        #print(t.size(), temb.size())
        
        temb = self.t_encoder(temb).view(-1, self.t_enc_dim, 1, 1).expand(-1, -1, 10, 10)
        
        xemb = self.con_x_encoder1(x)
        xemb = self.con_x_encoder2(xemb)
        xemb = self.con_x_encoder3(xemb)
        xemb = self.con_x_encoder4(xemb)
        
        if self.n_cond > 0:
            h = torch.cat([xemb, temb, selfcond, cond], 1)
        else:
            h = torch.cat([xemb, temb, selfcond], 1)
            
        out = self.con_x_decoder1(h)
        out = self.con_x_decoder2(out)
        out = self.con_x_decoder3(out)
        out = self.con_x_decoder4(out)

        return out


#----------------------------------------------------------------------------------------------------------#
#                                               CHEBYSHEV KAN                                              #
#----------------------------------------------------------------------------------------------------------#
# TypeError: unsupported operand type(s) for ** or pow(): 'tuple' and 'int'

class ChebyScoreKAN(torch.nn.Module):
    def __init__(self, encoder_layers=[256,256], pos_dim=128, decoder_layers=[256,256], x_dim=1, n_cond=1):

        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim *2
        self.locals = [encoder_layers, pos_dim, decoder_layers, x_dim]
        self.n_cond = n_cond


        self.net = KACN(layers_hidden = [3 * t_enc_dim + 1] + decoder_layers +[x_dim])
        self.t_encoder = KACN(layers_hidden = [pos_dim] + encoder_layers +[t_enc_dim])
        self.x_encoder = KACN(layers_hidden = [x_dim] + encoder_layers +[t_enc_dim])            
        self.e_encoder = KACN(layers_hidden = [1] + encoder_layers +[t_enc_dim])
        
        
    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if len(selfcond.shape) == 1:
            selfcond = selfcond.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        
        xemb = self.x_encoder(x)
        
            
        if self.n_cond > 0:
            eemb = cond
            eemb = self.e_encoder(eemb)
            h = torch.cat([xemb, temb, selfcond, eemb], -1)
        else:
            h = torch.cat([xemb ,temb, selfcond], -1)
                
        out = self.net(h) 
        return out

class ChebyScoreKANConv(torch.nn.Module):

    def __init__(self, encoder_layers=[256,256], temb_dim=128, conv_dof=32, x_dim=100, n_cond=0):
        super().__init__()
        self.conv_dof = conv_dof
        self.n_cond = n_cond
        
        self.temb_dim = temb_dim
        self.encoder_layers = encoder_layers
        self.t_enc_dim = self.conv_dof
        
        self.bias = False
        
        self.t_encoder = KACN(layers_hidden = [self.temb_dim] + self.encoder_layers +[self.t_enc_dim])

        self.con_x_encoder1 = KACNConv2DLayer(1, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_encoder2 = KACNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_encoder3 = KACNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_encoder4 = KACNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        
        
        self.con_x_decoder1 = KACNConv2DLayer(self.conv_dof*2+self.n_cond+1, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_decoder2 = KACNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_decoder3 = KACNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_decoder4 = KACNConv2DLayer(self.conv_dof, 1, 3, stride=(1, 1), padding=1)
 

    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if self.n_cond > 0:
            cond = cond.view(-1, self.n_cond, 1, 1).expand(-1, -1, 10, 10)

        temb = get_timestep_embedding(t, self.temb_dim)
        #print(t.size(), temb.size())
        
        temb = self.t_encoder(temb).view(-1, self.t_enc_dim, 1, 1).expand(-1, -1, 10, 10)
        
        xemb = self.con_x_encoder1(x)
        xemb = self.con_x_encoder2(xemb)
        xemb = self.con_x_encoder3(xemb)
        xemb = self.con_x_encoder4(xemb)
        
        if self.n_cond > 0:
            h = torch.cat([xemb, temb, selfcond, cond], 1)
        else:
            h = torch.cat([xemb, temb, selfcond], 1)
            
        out = self.con_x_decoder1(h)
        out = self.con_x_decoder2(out)
        out = self.con_x_decoder3(out)
        out = self.con_x_decoder4(out)

        return out


#----------------------------------------------------------------------------------------------------------#
#                                                  GRAM KAN                                                #
#----------------------------------------------------------------------------------------------------------#
# TypeError: randn(): argument 'size' must be tuple of ints, but found element of type tuple at pos 4

class GramScoreKAN(torch.nn.Module):
    def __init__(self, encoder_layers=[256,256], pos_dim=128, decoder_layers=[256,256], x_dim=1, n_cond=1):

        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim *2
        self.locals = [encoder_layers, pos_dim, decoder_layers, x_dim]
        self.n_cond = n_cond


        self.net = KAGN(layers_hidden = [3 * t_enc_dim + 1] + decoder_layers +[x_dim])
        self.t_encoder = KAGN(layers_hidden = [pos_dim] + encoder_layers +[t_enc_dim])
        self.x_encoder = KAGN(layers_hidden = [x_dim] + encoder_layers +[t_enc_dim])            
        self.e_encoder = KAGN(layers_hidden = [1] + encoder_layers +[t_enc_dim])
        
        
    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if len(selfcond.shape) == 1:
            selfcond = selfcond.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        
        xemb = self.x_encoder(x)
        
            
        if self.n_cond > 0:
            eemb = cond
            eemb = self.e_encoder(eemb)
            h = torch.cat([xemb, temb, selfcond, eemb], -1)
        else:
            h = torch.cat([xemb ,temb, selfcond], -1)
                
        out = self.net(h) 
        return out

class GramScoreKANConv(torch.nn.Module):

    def __init__(self, encoder_layers=[256,256], temb_dim=128, conv_dof=32, x_dim=100, n_cond=0):
        super().__init__()
        self.conv_dof = conv_dof
        self.n_cond = n_cond
        
        self.temb_dim = temb_dim
        self.encoder_layers = encoder_layers
        self.t_enc_dim = self.conv_dof
        
        self.bias = False
        
        self.t_encoder = KAGN(layers_hidden = [self.temb_dim] + self.encoder_layers +[self.t_enc_dim])

        self.con_x_encoder1 = KAGNConv2DLayer(1, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_encoder2 = KAGNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_encoder3 = KAGNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_encoder4 = KAGNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        
        
        self.con_x_decoder1 = KAGNConv2DLayer(self.conv_dof*2+self.n_cond+1, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_decoder2 = KAGNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_decoder3 = KAGNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_decoder4 = KAGNConv2DLayer(self.conv_dof, 1, 3, stride=1, padding=1)


    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if self.n_cond > 0:
            cond = cond.view(-1, self.n_cond, 1, 1).expand(-1, -1, 10, 10)

        temb = get_timestep_embedding(t, self.temb_dim)
        #print(t.size(), temb.size())
        
        temb = self.t_encoder(temb).view(-1, self.t_enc_dim, 1, 1).expand(-1, -1, 10, 10)
        
        xemb = self.con_x_encoder1(x)
        xemb = self.con_x_encoder2(xemb)
        xemb = self.con_x_encoder3(xemb)
        xemb = self.con_x_encoder4(xemb)
        
        if self.n_cond > 0:
            h = torch.cat([xemb, temb, selfcond, cond], 1)
        else:
            h = torch.cat([xemb, temb, selfcond], 1)
            
        out = self.con_x_decoder1(h)
        out = self.con_x_decoder2(out)
        out = self.con_x_decoder3(out)
        out = self.con_x_decoder4(out)

        return out


#----------------------------------------------------------------------------------------------------------#
#                                                 JACOBI KAN                                               #
#----------------------------------------------------------------------------------------------------------#
# TypeError: randn(): argument 'size' must be tuple of ints, but found element of type tuple at pos 4

class JacobiScoreKAN(torch.nn.Module):
    def __init__(self, encoder_layers=[256,256], pos_dim=128, decoder_layers=[256,256], x_dim=1, n_cond=1):

        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim *2
        self.locals = [encoder_layers, pos_dim, decoder_layers, x_dim]
        self.n_cond = n_cond


        self.net = KAJN(layers_hidden = [3 * t_enc_dim + 1] + decoder_layers +[x_dim])
        self.t_encoder = KAJN(layers_hidden = [pos_dim] + encoder_layers +[t_enc_dim])
        self.x_encoder = KAJN(layers_hidden = [x_dim] + encoder_layers +[t_enc_dim])            
        self.e_encoder = KAJN(layers_hidden = [1] + encoder_layers +[t_enc_dim])
        
        
    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if len(selfcond.shape) == 1:
            selfcond = selfcond.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        
        xemb = self.x_encoder(x)
        
            
        if self.n_cond > 0:
            eemb = cond
            eemb = self.e_encoder(eemb)
            h = torch.cat([xemb, temb, selfcond, eemb], -1)
        else:
            h = torch.cat([xemb ,temb, selfcond], -1)
                
        out = self.net(h) 
        return out

class JacobiScoreKANConv(torch.nn.Module):

    def __init__(self, encoder_layers=[256,256], temb_dim=128, conv_dof=32, x_dim=100, n_cond=0):
        super().__init__()
        self.conv_dof = conv_dof
        self.n_cond = n_cond
        
        self.temb_dim = temb_dim
        self.encoder_layers = encoder_layers
        self.t_enc_dim = self.conv_dof
        
        self.bias = False
        
        self.t_encoder = KAJN(layers_hidden = [self.temb_dim] + self.encoder_layers +[self.t_enc_dim])

        self.con_x_encoder1 = KAJNConv2DLayer(1, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_encoder2 = KAJNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_encoder3 = KAJNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_encoder4 = KAJNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        
        
        self.con_x_decoder1 = KAJNConv2DLayer(self.conv_dof*2+self.n_cond+1, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_decoder2 = KAJNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_decoder3 = KAJNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_decoder4 = KAJNConv2DLayer(self.conv_dof, 1, 3, stride=(1, 1), padding=1)


    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if self.n_cond > 0:
            cond = cond.view(-1, self.n_cond, 1, 1).expand(-1, -1, 10, 10)

        temb = get_timestep_embedding(t, self.temb_dim)
        #print(t.size(), temb.size())
        
        temb = self.t_encoder(temb).view(-1, self.t_enc_dim, 1, 1).expand(-1, -1, 10, 10)
        
        xemb = self.con_x_encoder1(x)
        xemb = self.con_x_encoder2(xemb)
        xemb = self.con_x_encoder3(xemb)
        xemb = self.con_x_encoder4(xemb)
        
        if self.n_cond > 0:
            h = torch.cat([xemb, temb, selfcond, cond], 1)
        else:
            h = torch.cat([xemb, temb, selfcond], 1)
            
        out = self.con_x_decoder1(h)
        out = self.con_x_decoder2(out)
        out = self.con_x_decoder3(out)
        out = self.con_x_decoder4(out)

        return out


#----------------------------------------------------------------------------------------------------------#
#                                                LAGRANGE KAN                                              #
#----------------------------------------------------------------------------------------------------------#
# TypeError: randn(): argument 'size' must be tuple of ints, but found element of type tuple at pos 4

class LagrangeScoreKAN(torch.nn.Module):
    def __init__(self, encoder_layers=[256,256], pos_dim=128, decoder_layers=[256,256], x_dim=1, n_cond=1):

        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim *2
        self.locals = [encoder_layers, pos_dim, decoder_layers, x_dim]
        self.n_cond = n_cond


        self.net = KALN(layers_hidden = [3 * t_enc_dim + 1] + decoder_layers +[x_dim])
        self.t_encoder = KALN(layers_hidden = [pos_dim] + encoder_layers +[t_enc_dim])
        self.x_encoder = KALN(layers_hidden = [x_dim] + encoder_layers +[t_enc_dim])            
        self.e_encoder = KALN(layers_hidden = [1] + encoder_layers +[t_enc_dim])
        
        
    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if len(selfcond.shape) == 1:
            selfcond = selfcond.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        
        xemb = self.x_encoder(x)
        
            
        if self.n_cond > 0:
            eemb = cond
            eemb = self.e_encoder(eemb)
            h = torch.cat([xemb, temb, selfcond, eemb], -1)
        else:
            h = torch.cat([xemb ,temb, selfcond], -1)
                
        out = self.net(h) 
        return out

class LagrangeScoreKANConv(torch.nn.Module):

    def __init__(self, encoder_layers=[256,256], temb_dim=128, conv_dof=32, x_dim=100, n_cond=0):
        super().__init__()
        self.conv_dof = conv_dof
        self.n_cond = n_cond
        
        self.temb_dim = temb_dim
        self.encoder_layers = encoder_layers
        self.t_enc_dim = self.conv_dof
        
        self.bias = False
        
        self.t_encoder = KALN(layers_hidden = [self.temb_dim] + self.encoder_layers +[self.t_enc_dim])

        self.con_x_encoder1 = KALNConv2DLayer(1, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_encoder2 = KALNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_encoder3 = KALNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_encoder4 = KALNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        
        
        self.con_x_decoder1 = KALNConv2DLayer(self.conv_dof*2+self.n_cond+1, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_decoder2 = KALNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_decoder3 = KALNConv2DLayer(self.conv_dof, self.conv_dof, 3, stride=1, padding=1)
        self.con_x_decoder4 = KALNConv2DLayer(self.conv_dof, 1, 3, stride=(1, 1), padding=1)


    def forward(self, x, t, cond=None, selfcond=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if self.n_cond > 0:
            cond = cond.view(-1, self.n_cond, 1, 1).expand(-1, -1, 10, 10)

        temb = get_timestep_embedding(t, self.temb_dim)
        #print(t.size(), temb.size())
        
        temb = self.t_encoder(temb).view(-1, self.t_enc_dim, 1, 1).expand(-1, -1, 10, 10)
        
        xemb = self.con_x_encoder1(x)
        xemb = self.con_x_encoder2(xemb)
        xemb = self.con_x_encoder3(xemb)
        xemb = self.con_x_encoder4(xemb)
        
        if self.n_cond > 0:
            h = torch.cat([xemb, temb, selfcond, cond], 1)
        else:
            h = torch.cat([xemb, temb, selfcond], 1)
            
        out = self.con_x_decoder1(h)
        out = self.con_x_decoder2(out)
        out = self.con_x_decoder3(out)
        out = self.con_x_decoder4(out)

        return out




