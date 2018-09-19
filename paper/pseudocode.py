while "Forever":
    w = 512 # mora biti 2^n
    img = read_camera() # int8[480,640,3]
    # uzmi samo zelenu komponentu
    img = img[:,:,1] # int8[480,640]
    img /= ref_profile # normalizacija 
    img = crop_resize(img, w) # int8[w,w]
    img = img.as(float32) # float32[w,w]
    # salji sliku na graficku karticu
    frame_gpu = to_gpu(img)
    # izracun amplitude na GPU
    frame_gpu = calc_amplitude(frame_gpu)
    # pretvori u kompleksni broj
    # complex64[w,w]
    field_gpu = add_phase(frame_gpu)
    # prebaci u frekvencijsku domenu
    field_gpu = fft(field_gpu)
    # konvolucija u frekv. domeni
    field_gpu = field_gpu * h_gpu
    # vrati u prostornu domenu
    field_gpu = ifft(field_gpu)
    # izracun intenziteta iz amplitude
    frame_gpu = abs2(field_gpu)
    result = to_cpu(frame_gpu)
    display(result)