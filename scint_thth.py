import numpy as np
import matplotlib as mpl
#mpl.use('TkAgg')
import matplotlib.pyplot as plt
import math
from ruamel.yaml import YAML
import os
from scipy.optimize import curve_fit

#class SS
#create curve function on SS

#create dot fitter

class thth():
    LightSpeed = 299792458. #m/s
    mas = 1./1000.*math.pi/648000. #mas in radians

    def __init__(self,file_specs,figure,ax,images,brightness):
        #load specifications
        yaml = YAML(typ='safe')
        with open(file_specs,'r') as readfile:
            specs = yaml.load(readfile)
        file_thth = specs['file_thth']
        file_thetas = specs['file_thetas']
        self.nu_half = specs['nu_half']
        self.veff = specs['veff']
        self.ax = ax
        self.figure = figure
        self.images = images
        self.brightness = brightness
        #load thth
        self.thth = np.load(file_thth)
        #load thetas
        self.thetas = np.load(file_thetas)
        self.N_th = len(self.thetas)
        self.thetas = -self.LightSpeed/self.nu_half/self.veff*self.thetas/self.mas
        #create logarithmic thth
        self.thth_log10 = np.copy(self.thth)
        min_nonzero = np.min(self.thth_log10[np.nonzero(self.thth_log10)])
        self.thth_log10[self.thth_log10 == 0] = min_nonzero
        self.thth_log10 = np.log10(self.thth_log10)
        
        #plot thth
        # - axis properties
        ax.set_xlim([float(np.min(self.thetas)),float(np.max(self.thetas))])
        ax.set_ylim([float(np.min(self.thetas)),float(np.max(self.thetas))])
        ax.set_title(r"$\theta$-$\theta$ diagram")
        ax.set_xlabel(r"$\theta_1$ [mas]")
        ax.set_ylabel(r"$\theta_2$ [mas]")
        # - colormesh specifications
        offset = (self.thetas[1]-self.thetas[0])/2.
        cmap = 'viridis'
        vmin = specs.get('thth_vmin',float(np.min(self.thth_log10)))
        vmax = specs.get('thth_vmax',float(np.max(self.thth_log10)))
        # - draw the plot
        self.plot_thth_grid = self.ax.pcolormesh((self.thetas-offset),(self.thetas-offset),np.swapaxes(self.thth_log10,0,1),cmap=cmap,vmin=vmin,vmax=vmax)
        self.figure.colorbar(self.plot_thth_grid, ax=self.ax)
        
        #plot thth curve
        th1,th2 = self.get_thth_curve(0.,0.,0.)
        self.plot_thth_curve, = self.ax.plot(th1,th2,color='red',linestyle='-',marker='o',markersize=0,alpha=0.5)
        
        #plot thth point
        points = self.images.get_points_array()
        self.plot_point, = self.ax.plot(points[0,0],points[0,1],color='red',linestyle='',marker='o',markersize=3, fillstyle='none',alpha=0.5)
        self.plot_points, = self.ax.plot(points[:,0],points[:,1],color='lightblue',linestyle='',marker='o',markersize=3, fillstyle='none',alpha=0.5)
        
        #plot found lines
        self.plot_lines = []
        self.update_lines()
        
        #plot brightness distributions
        self.update_mus()

    def get_thth_curve(self,theta,alpha,beta):
        #load data
        th_par = theta*np.cos(np.deg2rad(alpha))
        th_ort = theta*np.sin(np.deg2rad(alpha))
        om = np.tan(np.deg2rad(beta))
        #create thth coordinates
        th1 = 0.5*((self.thetas**2-th_par**2-th_ort**2)/(self.thetas-th_par-om*th_ort)+self.thetas-th_par-om*th_ort)
        th2 = 0.5*((self.thetas**2-th_par**2-th_ort**2)/(self.thetas-th_par-om*th_ort)-self.thetas+th_par+om*th_ort)
        data = [th1,th2]
        return data
        
    def get_lines(self):
        data = np.asarray(self.images.images)
        lines = np.empty((len(data),2,self.N_th),dtype=float)
        for index in range(len(data)):
            lines[index,:,:] = np.array(self.get_thth_curve(data[index,0],data[index,1],data[index,2]))
        return lines
        
    def update_thth_curve(self,theta,alpha,beta):
        th1,th2 = self.get_thth_curve(theta,alpha,beta)
        self.plot_thth_curve.set_xdata(th1)
        self.plot_thth_curve.set_ydata(th2)
        self.figure.canvas.draw_idle()
        
    def update_lines(self):
        lines = self.get_lines()
        N_lines = len(self.images.images)
        N_plots = len(self.plot_lines)
        for index in range(N_lines):
            if index < N_plots:
                self.plot_lines[index].set_xdata(lines[index,0,:])
                self.plot_lines[index].set_ydata(lines[index,1,:])
            else:
                self.plot_lines.append(self.ax.plot(lines[index,0,:],lines[index,1,:],color='lightblue',linestyle='-', fillstyle='none',alpha=0.5,linewidth=1)[0])
                #self.plot_lines[index] = self.ax.plot(lines[index,0,:],lines[index,1,:],color='lightblue',linestyle='-', fillstyle='none',alpha=0.5,linewidth=1)
        if N_lines<N_plots:
            for index in range(N_lines,N_plots):
                self.plot_lines[index].remove()
            del self.plot_lines[N_lines:N_plots]
        self.figure.canvas.draw_idle()
        
    def create_widgets(self,xmin,ymin,xmax,ymax):
        #compute locations
        nrows = 7
        ncols = 3
        yspace = 0.02
        xspace = 0.02
        xfullwidth = xmax-xmin
        yfullwidth = ymax-ymin
        xwidth = (xfullwidth-(ncols-1)*yspace)/ncols
        ywidth = (yfullwidth-(nrows-1)*yspace)/nrows
        xpos = [xmin+i*(xwidth+xspace) for i in range(ncols)]
        ypos = [ymin+i*(ywidth+yspace) for i in range(nrows)]
        #load data
        theta,alpha,beta = self.images.get_image_array()
        self.update_thth_curve(theta,alpha,beta)
        points = self.images.get_points_array()
        #initialize widgets for the curve
        self.slider_th = mpl.widgets.Slider(plt.axes([xpos[0],ypos[2],xfullwidth,ywidth]),r'$\theta$',-30.,30.,valinit=theta)
        self.slider_alpha = mpl.widgets.Slider(plt.axes([xpos[0],ypos[1],xfullwidth,ywidth]),r'$\alpha$',-90.,90.,valinit=alpha)
        self.slider_beta = mpl.widgets.Slider(plt.axes([xpos[0],ypos[0],xfullwidth,ywidth]),r'$\beta$',-90.,90.,valinit=beta)
        self.button_reset = mpl.widgets.Button(plt.axes([xpos[2],ypos[3],xwidth,ywidth]), 'Reset')
        self.button_anchor = mpl.widgets.Button(plt.axes([xpos[1],ypos[3],xwidth,ywidth]), 'Anchor')
        self.box_th_range = mpl.widgets.TextBox(plt.axes([xpos[0],ypos[3],xwidth,ywidth]), r'$\Delta\theta$', initial="60.0")
        #initialize widgets for the images
        self.box_th1 = mpl.widgets.TextBox(plt.axes([xpos[0],ypos[4],xwidth,ywidth]), r'$\theta_1$', initial=str(points[0,0]))
        self.box_th2 = mpl.widgets.TextBox(plt.axes([xpos[1],ypos[4],xwidth,ywidth]), r'$\theta_2$', initial=str(points[0,1]))
        self.button_fit_points = mpl.widgets.Button(plt.axes([xpos[2],ypos[4],xwidth,ywidth]), r'fit points')
        self.box_point_number = mpl.widgets.TextBox(plt.axes([xpos[0],ypos[5],xwidth,ywidth]), r'#p', initial="0")
        self.button_delete_point = mpl.widgets.Button(plt.axes([xpos[1],ypos[5],xwidth,ywidth]), r'delete point')
        self.button_save_point = mpl.widgets.Button(plt.axes([xpos[2],ypos[5],xwidth,ywidth]), r'save point')
        self.box_line_number = mpl.widgets.TextBox(plt.axes([xpos[0],ypos[6],xwidth,ywidth]), r'#l', initial="0")
        self.button_delete_line = mpl.widgets.Button(plt.axes([xpos[1],ypos[6],xwidth,ywidth]), r'delete line')
        self.button_save_line = mpl.widgets.Button(plt.axes([xpos[2],ypos[6],xwidth,ywidth]), r'save line')
        #tell widgets what to do on use
        self.slider_th.on_changed(self.update_curve_widget_values)
        self.slider_alpha.on_changed(self.update_curve_widget_values)
        self.slider_beta.on_changed(self.update_curve_widget_values)
        self.button_reset.on_clicked(self.reset_curve_widget_values)
        self.button_anchor.on_clicked(self.anchor_curve_widget_values)
        self.box_th_range.on_submit(self.rescale_curve_widget_th)
        self.box_th1.on_submit(self.update_image_widget_th1)
        self.box_th2.on_submit(self.update_image_widget_th2)
        self.button_save_point.on_clicked(self.submit_point)
        self.box_point_number.on_submit(self.change_point_number)
        self.button_delete_point.on_clicked(self.delete_point)
        self.button_fit_points.on_clicked(self.fit_thth_curve)
        self.box_line_number.on_submit(self.change_line_number)
        self.button_delete_line.on_clicked(self.delete_line)
        self.button_save_line.on_clicked(self.submit_line)
        
        #add point position by clicking
        self.figure.canvas.mpl_connect('button_press_event', self.onclick)
        
        #update plots
        self.update_mu()
        
    def onclick(self,event):
        if str(event.button)=='MouseButton.RIGHT':
            th1 = event.xdata
            th2 = event.ydata
            self.box_th1.set_val('{:.4f}'.format(th1))
            self.box_th2.set_val('{:.4f}'.format(th2))
            self.plot_point.set_xdata(th1)
            self.plot_point.set_ydata(th2)
            self.figure.canvas.draw_idle()
        
    def update_curve_widget_values(self,event):
        self.update_thth_curve(self.slider_th.val,self.slider_alpha.val,self.slider_beta.val)
        self.images.update_image_plot(self.slider_th.val,self.slider_alpha.val)
        self.update_mu()
        
    def reset_curve_widget_values(self,event):
        self.slider_th.reset()
        self.slider_alpha.reset()
        self.slider_beta.reset()
        
    def anchor_curve_widget_values(self,event):
        self.slider_th.valinit = self.slider_th.val
        self.slider_alpha.valinit = self.slider_alpha.val
        self.slider_beta.valinit = self.slider_beta.val
        
    def rescale_curve_widget_th(self,text):
        mid = self.slider_th.valinit
        range_th = float(text)
        self.slider_th.valmin = mid-range_th/2.
        self.slider_th.valmax = mid+range_th/2.
        self.slider_th.ax.set_xlim(self.slider_th.valmin,self.slider_th.valmax)
        
    def update_image_widget_th1(self,text):
        th1 = float(text)
        self.plot_point.set_xdata(th1)
        self.figure.canvas.draw_idle()
        
    def update_image_widget_th2(self,text):
        th2 = float(text)
        self.plot_point.set_ydata(th2)
        self.figure.canvas.draw_idle()
        
    def submit_point(self,event):
        number = self.images.enter_point(float(self.box_th1.text),float(self.box_th2.text))
        self.images.save_data()
        self.box_point_number.set_val(number)
        points = self.images.get_points_array()
        self.plot_points.set_xdata(points[:,0])
        self.plot_points.set_ydata(points[:,1])
        self.figure.canvas.draw_idle()
        
    def delete_point(self,event):
        number = self.images.delete_point()
        self.images.save_data()
        self.box_point_number.set_val(number)
        points = self.images.get_points_array()
        self.plot_points.set_xdata(points[:,0])
        self.plot_points.set_ydata(points[:,1])
        self.figure.canvas.draw_idle()
        
    def change_point_number(self,text):
        number = int(text)
        new_text = self.images.change_point(number)
        self.box_point_number.set_val(str(new_text))
        number = int(new_text)
        points = self.images.get_points_array()
        self.plot_point.set_xdata(points[number,0])
        self.plot_point.set_ydata(points[number,1])
        self.box_th1.set_val(points[number,0])
        self.box_th2.set_val(points[number,1])
        self.figure.canvas.draw_idle()
        
    def _fitfunc_thth_curve(self,th1,theta,alpha,beta):
        #load data
        th_par = theta*np.cos(np.deg2rad(alpha))
        th_ort = theta*np.sin(np.deg2rad(alpha))
        om = np.tan(np.deg2rad(beta))
        #create thth coordinates
        theta_a = 0.5*(th1+th_par+om*th_ort) + np.sqrt(0.25*(th1+th_par+om*th_ort)**2-th1*(th_par+om*th_ort)+(1.-om**2)/2.*th_ort**2-om*th_par*th_ort)
        theta_b = 0.5*(th1+th_par+om*th_ort) - np.sqrt(0.25*(th1+th_par+om*th_ort)**2-th1*(th_par+om*th_ort)+(1.-om**2)/2.*th_ort**2-om*th_par*th_ort)
        #th1 = 0.5*((self.thetas**2-th_par**2-th_ort**2)/(self.thetas-th_par-om*th_ort)+self.thetas-th_par-om*th_ort)
        th2_a = 0.5*((theta_a**2-th_par**2-th_ort**2)/(theta_a-th_par-om*th_ort)-theta_a+th_par+om*th_ort)
        th2_b = 0.5*((theta_b**2-th_par**2-th_ort**2)/(theta_b-th_par-om*th_ort)-theta_b+th_par+om*th_ort)
        th2 = np.empty(th1.shape,dtype=float)
        for i_th1 in range(len(th1)):
            th2_ab = np.array([th2_a[i_th1],th2_b[i_th1]])
            index = np.nanargmin(np.abs(th2_ab-theta))
            th2[i_th1] = th2_ab[index]
        return th2
        
    def fit_thth_curve(self,event):
        theta,alpha,beta = self.images.fit_thth_curve(self._fitfunc_thth_curve,self.slider_th.val,self.slider_alpha.val,self.slider_beta.val)
        self.slider_th.set_val(theta)
        self.slider_alpha.set_val(alpha)
        self.slider_beta.set_val(beta)
        
    def change_line_number(self,text):
        number = int(text)
        i_line,i_point = self.images.change_line(number)
        self.box_line_number.set_val(str(i_line))
        points = self.images.get_points_array()
        self.plot_point.set_xdata(points[i_point,0])
        self.plot_point.set_ydata(points[i_point,1])
        self.box_th1.set_val(points[i_point,0])
        self.box_th2.set_val(points[i_point,1])
        self.plot_points.set_xdata(points[:,0])
        self.plot_points.set_ydata(points[:,1])
        theta,alpha,beta = self.images.get_image_array()
        self.update_thth_curve(theta,alpha,beta)
        self.update_lines()
        self.slider_th.set_val(theta)
        self.slider_alpha.set_val(alpha)
        self.slider_beta.set_val(beta)
        self.update_mu()
        self.figure.canvas.draw_idle()
        
    def delete_line(self,event):
        i_line,i_point = self.images.delete_line()
        self.images.save_data()
        self.change_line_number(str(i_line))
        self.change_point_number(i_point)
        points = self.images.get_points_array()
        self.plot_points.set_xdata(points[:,0])
        self.plot_points.set_ydata(points[:,1])
        self.plot_point.set_xdata(points[i_point,0])
        self.plot_point.set_ydata(points[i_point,1])
        self.box_th1.set_val(points[i_point,0])
        self.box_th2.set_val(points[i_point,1])
        self.update_mus()
        self.figure.canvas.draw_idle()
        
    def submit_line(self,event):
        i_line,i_point = self.images.enter_line(self.slider_th.val,self.slider_alpha.val,self.slider_beta.val)
        self.images.save_data()
        self.change_point_number(i_point)
        points = self.images.get_points_array()
        self.plot_points.set_xdata(points[:,0])
        self.plot_points.set_ydata(points[:,1])
        self.plot_point.set_xdata(points[i_point,0])
        self.plot_point.set_ydata(points[i_point,1])
        self.box_th1.set_val(points[i_point,0])
        self.box_th2.set_val(points[i_point,1])
        line = self.images.get_image_array()
        self.change_line_number(str(i_line))
        self.slider_th.set_val(line[0])
        self.slider_alpha.set_val(line[1])
        self.slider_beta.set_val(line[2])
        self.update_mus()
        self.figure.canvas.draw_idle()
        
    def update_mu(self):
        theta,alpha,beta = self.images.get_image_array()
        th1,th2 = self.get_thth_curve(theta,alpha,beta)
        minth = np.min(self.thetas)
        maxth = np.max(self.thetas)
        spanth = maxth-minth
        brightness = np.zeros(self.N_th,dtype=float)
        i1 = np.zeros(self.N_th,dtype=int)
        i2 = np.zeros(self.N_th,dtype=int)
        for i in range(self.N_th):
            i1[i] = self.N_th-int((th1[i]-minth)/spanth*self.N_th+0.5)
            i2[i] = self.N_th-int((th2[i]-minth)/spanth*self.N_th+0.5)
            try:
                brightness[i] = self.thth_log10[i1[i],i2[i]]
            except:
                pass
        self.brightness.update_mu_plot(brightness)
        
    def update_mus(self):
        lines = np.asarray(self.images.images)
        N_lines = len(lines)
        th1 = np.empty((N_lines,self.N_th),dtype=float)
        th2 = np.empty((N_lines,self.N_th),dtype=float)
        for index in range(N_lines):
            th1[index,:],th2[index,:] = self.get_thth_curve(lines[index,0],lines[index,1],lines[index,2])
        minth = np.min(self.thetas)
        maxth = np.max(self.thetas)
        spanth = maxth-minth
        brightness = np.zeros((N_lines,self.N_th),dtype=float)
        for i_line in range(N_lines):
            i1 = np.zeros(self.N_th,dtype=int)
            i2 = np.zeros(self.N_th,dtype=int)
            for i_th in range(self.N_th):
                i1[i_th] = self.N_th-int((th1[i_line,i_th]-minth)/spanth*self.N_th+0.5)
                i2[i_th] = self.N_th-int((th2[i_line,i_th]-minth)/spanth*self.N_th+0.5)
                try:
                    brightness[i_line,i_th] = self.thth_log10[i1[i_th],i2[i_th]]
                except:
                    pass
        self.brightness.update_mus_plot(brightness)
        
class images():
    LightSpeed = 299792458. #m/s
    mas = 1./1000.*math.pi/648000. #mas in radians
    
    def __init__(self,file_specs,figure,ax):
        #initialize data
        self.i_line = 0
        self.i_point = 0
        self.matrix = [[[0.,0.]]]
        self.images = [[0.,0.,0.]]
        yaml = YAML(typ='safe')
        with open(file_specs,'r') as readfile:
            specs = yaml.load(readfile)
        self.nu_half = specs['nu_half']
        self.veff = specs['veff']
        file_thetas = specs['file_thetas']
        self.file_points = specs['file_points']
        self.file_lines = specs['file_lines']
        if os.path.exists(self.file_points) and os.path.exists(self.file_lines):
            self.matrix = np.load(self.file_points,allow_pickle=True).tolist()
            self.images = np.load(self.file_lines,allow_pickle=True).tolist()
        self.N_lines = len(self.matrix)
        self.N_points = len(self.matrix[self.i_line])
        self.thetas = np.load(file_thetas)
        self.N_th = len(self.thetas)
        self.thetas = -self.LightSpeed/self.nu_half/self.veff*self.thetas/self.mas
        self.ax = ax
        self.figure = figure
        #check compatibility
        if self.N_lines<len(self.images):
            number = len(self.images)
            del self.images[self.N_lines:number]
        
        #initialize plot
        self.ax.set_xlim([np.min(self.thetas),np.max(self.thetas)])
        self.ax.set_ylim([np.min(self.thetas),np.max(self.thetas)])
        self.ax.set_xlabel(r"$\theta_x$ [mas]")
        self.ax.set_ylabel(r"$\theta_y$ [mas]")
        self.ax.set_title(r'screen')
        self.ax.axvline(0.,color='black',alpha=0.3)
        th_par,th_ort = self.get_images_pos_array()
        self.plot_images, = self.ax.plot(th_ort,th_par,color='black',linestyle='',marker='o',markersize=2,alpha=0.5)
        self.plot_image, = self.ax.plot(th_ort[self.i_line],th_par[self.i_line],color='red',linestyle='',marker='o',markersize=3)
        
    def save_data(self):
        np.save(self.file_points,np.array(self.matrix))
        np.save(self.file_lines,np.array(self.images))
        
    def fit_thth_curve(self,fitfunc,theta,alpha,beta):
        data = np.array(self.matrix[self.i_line])
        popt, pcov = curve_fit(fitfunc,data[:,0],data[:,1],p0=[theta,alpha,beta],bounds=([np.min(self.thetas),-90.,-90.],[np.max(self.thetas),90.,90.]))
        perr = np.sqrt(np.diag(pcov))
        fit_theta = popt[0]
        fit_alpha = popt[1]
        fit_beta = popt[2]
        return (fit_theta,fit_alpha,fit_beta)
        
    def enter_point(self,thx,thy):
        self.matrix[self.i_line][self.i_point] = [thx,thy]
        i_point = self.i_point + 1
        self.change_point(i_point)
        return self.i_point
        
    def change_point(self,i_point):
        if i_point < self.N_points:
            self.i_point = i_point
        else:
            self.N_points += 1
            self.i_point = self.N_points - 1
            self.matrix[self.i_line].append([0.,0.])
        return self.i_point
            
    def delete_point(self):
        if not (self.N_points==1):
            del self.matrix[self.i_line][self.i_point]
            self.N_points -= 1
        if self.i_point==self.N_points:
            self.i_point -= 1
        return self.i_point
        
    def enter_line(self,theta,alpha,beta):
        self.images[self.i_line] = [theta,alpha,beta]
        i_line = self.i_line + 1
        self.change_line(i_line)
        return (self.i_line,self.i_point)
        
    def change_line(self,i_line):
        if i_line < self.N_lines:
            self.i_line = i_line
        else:
            self.N_lines += 1
            self.i_line = self.N_lines - 1
            data = np.asarray(self.images)
            beta = np.mean(data[:,2])
            self.images.append([0.,0.,beta])
            self.matrix.append([[0.,0.]])
        self.N_points = len(self.matrix[self.i_line])
        self.i_point = self.N_points - 1
        th_par,th_ort = self.get_images_pos_array()
        self.plot_images.set_xdata(th_ort)
        self.plot_images.set_ydata(th_par)
        return (self.i_line,self.i_point)
        
    def delete_line(self):
        del self.matrix[self.i_line]
        del self.images[self.i_line]
        self.N_lines -= 1
        if self.i_line==self.N_lines:
            self.i_line -= 1
        self.N_points = len(self.matrix[self.i_line])
        self.i_point = self.N_points - 1
        return (self.i_line,self.i_point)
        
    def update_image_plot(self,theta,alpha):
        th_par = theta*np.cos(np.deg2rad(alpha))
        th_ort = theta*np.sin(np.deg2rad(alpha))
        self.plot_image.set_xdata(th_ort)
        self.plot_image.set_ydata(th_par)
        self.figure.canvas.draw_idle()
        
    def get_images_pos_array(self):
        data = np.asarray(self.images)
        theta = np.array(data[:,0])
        alpha = np.array(data[:,1])
        th_par = theta*np.cos(np.deg2rad(alpha))
        th_ort = theta*np.sin(np.deg2rad(alpha))
        return (th_par,th_ort)
        
    def get_points_array(self):
        points = np.asarray(self.matrix[self.i_line])
        return points
        
    def get_image_array(self):
        line = np.asarray(self.images[self.i_line])
        return line
        
class brightness():
    LightSpeed = 299792458. #m/s
    mas = 1./1000.*math.pi/648000. #mas in radians
    
    def __init__(self,file_specs,figure,ax):
        #initialize data
        yaml = YAML(typ='safe')
        with open(file_specs,'r') as readfile:
            specs = yaml.load(readfile)
        self.nu_half = specs['nu_half']
        self.veff = specs['veff']
        file_thetas = specs['file_thetas']
        # self.file_points = specs['file_points']
        # self.file_lines = specs['file_lines']
        self.thetas = np.load(file_thetas)
        self.N_th = len(self.thetas)
        self.thetas = -self.LightSpeed/self.nu_half/self.veff*self.thetas/self.mas
        self.ax = ax
        self.figure = figure
        
        #initialize plot
        self.ax.set_xlim([np.min(self.thetas),np.max(self.thetas)])
        self.ax.set_ylim([specs.get('thth_vmin',None),specs.get('thth_vmax',None)])
        self.ax.set_xlabel(r"$\theta$ [mas]")
        self.ax.set_ylabel(r"$\mu$")
        self.ax.set_title(r'brightness $\mu$')
        self.plot_mus = []
        self.plot_median, = self.ax.plot(self.thetas,np.zeros(self.thetas.shape,dtype=float),color='black',alpha=0.8,label="median")
        self.plot_brightness, = self.ax.plot(self.thetas,np.zeros(self.thetas.shape,dtype=float),color='red',alpha=0.8,label="current")
        self.ax.legend()
        
    def update_mu_plot(self,brightness):
        self.plot_brightness.set_ydata(brightness)
        self.figure.canvas.draw_idle()
        
    def update_mus_plot(self,mus):
        N_mu = len(mus)
        N_plots = len(self.plot_mus)
        for index in range(N_mu):
            if index < N_plots:
                self.plot_mus[index].set_ydata(mus[index,:])
            else:
                self.plot_mus.append(self.ax.plot(self.thetas,mus[index,:],color='lightblue',linestyle='-',alpha=0.,linewidth=1)[0])
        if N_mu<N_plots:
            for index in range(N_mu,N_plots):
                self.plot_mus[index].remove()
            del self.plot_mus[N_mu:N_plots]
        if not N_mu==0:
            mus_median = np.median(mus,axis=0)
            self.plot_median.set_ydata(mus_median)
        self.figure.canvas.draw_idle()
    
if __name__ == "__main__":
    #set up the canvas
    plot_width = 1600
    plot_height = 900
    plot_dpi = 100
    plot_bottom = 0.05
    plot_top = 0.95
    plot_left = 0.06
    plot_right = 0.75
    plot_wspace = 0.2
    plot_hspace = 0.2
    # - create figure and axes
    figure = plt.figure(figsize=(plot_width/plot_dpi,plot_height/plot_dpi),dpi=plot_dpi)
    plt.subplots_adjust(bottom=plot_bottom,top=plot_top,left=plot_left,right=plot_right,wspace=plot_wspace,hspace=plot_hspace)
    ax_thth = figure.add_subplot(2,1,1)
    ax_mu = figure.add_subplot(2,2,3)
    ax_screen = figure.add_subplot(2,2,4)
    
    #initialize plots
    file_specs = "specs.yaml"
    plot_images = images(file_specs,figure,ax_screen)
    plot_brightness = brightness(file_specs,figure,ax_mu)
    plot_thth = thth(file_specs,figure,ax_thth,plot_images,plot_brightness)
    plot_thth.create_widgets(0.71, 0.55, 0.95, 0.90)
    
    #show the fitter and tidy up
    plt.show()
    plt.clf()