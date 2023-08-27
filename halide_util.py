import halide as hl
import struct

def find_gpu_target():
    # Start with a target suitable for the machine you're running this on.
    target = hl.get_host_target()

    features_to_try = []
    if target.os == hl.TargetOS.Windows:
        # Try D3D12 first; if that fails, try OpenCL.
        if struct.calcsize("P") == 8:
            # D3D12Compute support is only available on 64-bit systems at present.
            features_to_try.append(hl.TargetFeature.D3D12Compute)
        features_to_try.append(hl.TargetFeature.OpenCL)
    elif target.os == hl.TargetOS.OSX:
        # OS X doesn't update its OpenCL drivers, so they tend to be broken.
        # CUDA would also be a fine choice on machines with NVidia GPUs.
        features_to_try.append(hl.TargetFeature.Metal)
    else:
        features_to_try.append(hl.TargetFeature.OpenCL)

    # Uncomment the following lines to also try CUDA:
    # features_to_try.append(hl.TargetFeature.CUDA);
    for f in features_to_try:
        new_target = target.with_feature(f)
        if (hl.host_supports_target_device(new_target)):
            return new_target

    print("Requested GPU(s) are not supported. (Do you have the proper hardware and/or driver installed?)")
    return target
