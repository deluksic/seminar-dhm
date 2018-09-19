#define _USE_MATH_DEFINES
#include <cuComplex.h>
#include <math.h>

__host__ __device__ static __inline__ cuComplex cx_exp(float real, float imag)
{
    float s, c;
    float e = expf(real);
    sincosf(imag, &s, &c);
    return make_cuComplex(c * e, s * e);
}

__host__ __device__ static __inline__ cuComplex cx_amp_ph(float amp, float ph)
{
    float s, c;
    sincosf(ph, &s, &c);
    return make_cuComplex(c * amp, s * amp);
}

__host__ __device__ static __inline__ float cuPhasef(cuComplex c)
{
    return atan2(c.y, c.x);
}

__host__ __device__ static __inline__ float cuCabs2f(cuComplex c)
{
    return c.x * c.x + c.y * c.y;
}

__global__ void propagator(cuComplex *p, float lambda, float side, float z)
{
    int w = gridDim.x * 2;
    int w1 = w - 1;
    int idx1 = blockIdx.x + blockIdx.y * w;
    int idx2 = w1 - blockIdx.x + blockIdx.y * w;
    int idx3 = blockIdx.x + (w1 - blockIdx.y) * w;
    int idx4 = w1 - blockIdx.x + (w1 - blockIdx.y) * w;
    float al = lambda / side * blockIdx.y;
    float bl = lambda / side * blockIdx.x;
    float al2bl2 = al * al + bl * bl;
    if (al2bl2 <= 1)
    {
        float arg = -2 * M_PI * z * sqrt(1 - al2bl2) / lambda;
        p[idx1] = p[idx2] = p[idx3] = p[idx4] = cx_amp_ph(1, arg);
    }
    else
    {
        p[idx1] = p[idx2] = p[idx3] = p[idx4] = make_cuComplex(0, 0);
    }
}

__global__ void apply_mask(cuComplex *dest, cuComplex *src, cuComplex *mask)
{
    int idx = blockIdx.x + blockIdx.y * gridDim.x;
    dest[idx] = cuCmulf(src[idx], mask[idx]);
}

__global__ void apply_conj_mask(cuComplex *dest, cuComplex *src, cuComplex *mask)
{
    int idx = blockIdx.x + blockIdx.y * gridDim.x;
    dest[idx] = cuCmulf(src[idx], cuConjf(mask[idx]));
}

__global__ void apply_phase(cuComplex *dest, float *amp, float *phase)
{
    int idx = blockIdx.x + blockIdx.y * gridDim.x;
    dest[idx] = cx_amp_ph(amp[idx], phase[idx]);
}

__global__ void abs2(float *dest, cuComplex *src)
{
    int idx = blockIdx.x + blockIdx.y * gridDim.x;
    dest[idx] = cuCabs2f(src[idx]);
}

__global__ void cx_to_phase(float *dest, cuComplex *src)
{
    int idx = blockIdx.x + blockIdx.y * gridDim.x;
    dest[idx] = cuPhasef(src[idx]);
}

__global__ void amp_from_intensity(float *dest, float *src)
{
    int idx = blockIdx.x + blockIdx.y * gridDim.x;
    dest[idx] = sqrtf(src[idx]);
}

__global__ void filter_obj_dom(cuComplex *t)
{
    int idx = blockIdx.x + blockIdx.y * gridDim.x;
    float am = cuCabsf(t[idx]);
    float ph = cuPhasef(t[idx]);
    if (am > 1)
    {
        t[idx].x = 1;
        t[idx].y = 0;
    }
}