from tkinter import *
from PIL import ImageTk, Image

class processImage:
    def __init__(self, filename):
        self.img = Image.open(filename)
        self.root = Tk()
        self.label = None
        
        
    def showImage(self):
        
        #converter a imagem a ser mostrada
        img_r = self.img.resize((1280,720))
        self.img = img_r
        img_tk = ImageTk.PhotoImage(img_r)
        print(img_tk)
        
        self.root.geometry("1280x720")
        #criar widget de exibição
        self.label = Label(self.root, image=img_tk)
        self.label.bind("<Button 1>", self.click_zoom)
        self.label.pack()
        self.root.mainloop()
        
    def click_zoom(self, event):
        x = event.x
        y = event.y
        
        #dar zoom
        img_origin = self.img
        img_cut = img_origin.crop((x-200, y-100, x+200, y+100))
        img = img_cut.resize((img_cut.size[0] * 2, img_cut.size[1]*2))
         
        #converter a imagem a ser mostrada
        img_tk = ImageTk.PhotoImage(img)
        
        print(img_tk)
            
        #exibir tela com o zoom
        root_z = Toplevel()
        root_z.geometry("400x200")
        label_z = Label(root_z, image=img_tk)
        label_z.pack()
        root_z.mainloop()
        
            

# Carrega a imagem
# img = Image.open("./mamografias/DleftCC/d_left_cc (1).png")
# img = img.resize((400, 400))

zoom = processImage("./mamografias/DleftCC/d_left_cc (1).png")
zoom.showImage()



