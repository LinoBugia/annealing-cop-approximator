from Funcs_Optimizers import *
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
# Ising model approximator
from Funcs_pbfGenerators import Generate_2D_Ising_pbf
import imageio
if __name__ == "__main__":
    #parameters
    num_MC = 1
    steps=50000
    
    number_particles = 80
    J=-1
    h=0
    degree = 2

    cooling_param=("constant",1,1)
    cooling_param=("logarithmic_step",1,1)

    Offset_increase = 100

    # Generate Ising Schedule:
    dim = "2D"
    
    seed_gen = 76243
    seed_rand =[]
    random.seed(seed_gen)
    for i in range(num_MC):
        seed_rand.append(random.uniform(0,1000))
        
    print("starting feature generation")
    start_time_gen =time.time() 
    print("variables = " + str(number_particles))
    
    pbf,trans_dict,inv_trans_dict = Generate_2D_Ising_pbf("2D",number_particles,J,h)
    
    pbf=Sort_pbf(pbf,2)
    pbf_var_dict = createPolyDict(pbf,pow(number_particles,2)) 
       
    
    print(evalPBF_List(pbf,[0,1]*round(pow(number_particles,2)/2)) ,evalPBF_List(pbf,[1]*round(pow(number_particles,2))) )
    print("feature generation time:" + str(time.time() - start_time_gen))
    
    #a,b,Trajectories_1sa,result_List_1sa=pbf_min_solver(pbf,pbf_var_dict,"simulatedAnnealing",steps,num_MC,cooling_param,seed_rand,seed_gen,visual_inst=False,offset_increase_rate=Offset_increase,save_addinfo=True, save_csv=True,random_start=True,Ising = True)   
    a,b,Trajectories_1da,result_List_1da=pbf_min_solver(pbf,pbf_var_dict,"digitalAnnealing",steps,num_MC,cooling_param,seed_rand,seed_gen,visual_inst=False,offset_increase_rate=Offset_increase,save_addinfo=True, save_csv=True,random_start=True,Ising = True)  
    print(len(result_List_1da))

    #pbf,trans_dict,inv_trans_dict = Generate_2D_Ising_pbf("2D",number_particles,-1,3)
    
    #pbf=Sort_pbf(pbf,2)
    #pbf_var_dict = createPolyDict(pbf,pow(number_particles,2)) 
       
    
    #print(evalPBF_List(pbf,[0,1]*round(pow(number_particles,2)/2)) ,evalPBF_List(pbf,[1]*round(pow(number_particles,2))) )
    #print("feature generation time:" + str(time.time() - start_time_gen))
    
    #a,b,Trajectories_2sa,result_List_2sa=pbf_min_solver(pbf,pbf_var_dict,"simulatedAnnealing",steps,10,cooling_param,seed_rand,seed_gen,visual_inst=True,offset_increase_rate=Offset_increase,save_addinfo=True, save_csv=False,random_start=True,Ising = True)   
    #a,b,Trajectories_2da,result_List_2da=pbf_min_solver(pbf,pbf_var_dict,"digitalAnnealing",steps,10,cooling_param,seed_rand,seed_gen,visual_inst=True,offset_increase_rate=Offset_increase,save_addinfo=True, save_csv=False,random_start=True,Ising = True)  
    #print(result_List)
    #Plot_Trajec_nice([Trajectories_1sa,Trajectories_1da])
    #Plot_Trajec_nice([Trajectories_2sa,Trajectories_2da])
    #print(result_List)
    Output =[] 
    if 1:

    
    
        #validate  
        list=[] 
        outputfolder = "/Users/lino/Documents/python/AnnealingCopApproximator/Code/IsingRunDA50"
        for k in range(len(result_List_1da)):
            
            if k>29999:
                print(k)
                # Create a sample 2D binary image (e.g., 10x10 grid of 0s and 1s)
                arr = np.random.randint(0, 2, size=(number_particles, number_particles))
                for i in range(pow(number_particles,2)):
                    arr[trans_dict[i]] = result_List_1da[k][i]
                # Create color image from 0/1 array
                # Colors for 0 and 1
                color0 = (255, 255, 255)  # white
                color1 = (0, 0, 0)        # black
                img_data = np.zeros((number_particles, number_particles, 3), dtype=np.uint8)
                img_data[arr == 0] = color0
                img_data[arr == 1] = color1

                # Convert to PIL Image
                img = Image.fromarray(img_data)

                # Optional: scale up so it’s more visible (e.g., each pixel = 20×20 block)
                square = img.resize((number_particles*50, number_particles*50), resample=Image.NEAREST)

                # ✅ Use a system font on macOS
                font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
                font = ImageFont.truetype(font_path, size=int(number_particles*2))
                # Header settings
                header_height = 500
                header_color = (50, 50, 50)   # dark gray
                text_color = (255, 255, 255)  # white
                header_text = f"DA Num:{k:08d} "+"Energy:" +str( Trajectories_1da[0][k])

                # Create new image (taller: header + original)
                new_height = number_particles*50 + header_height
                new_img = Image.new("RGB", (number_particles*50, new_height), color=(255, 255, 255))

                # Paste header + original
                draw = ImageDraw.Draw(new_img)
                draw.rectangle([0, 0, number_particles*50, header_height], fill=header_color)
                new_img.paste(square, (0, header_height))

                # Load font

                #font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=200)


                # --- FIX: use textbbox() instead of textsize() ---
                bbox = draw.textbbox((0, 0), header_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]

                # Center the text
                text_x = (number_particles*50 - text_w) // 2
                text_y = (header_height - text_h) // 2
                draw.text((text_x, text_y), header_text, fill=text_color, font=font)
                filename = f"img{k:08d}.png"
                new_img.save(os.path.join(outputfolder,filename))