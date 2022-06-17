// Generate the work group ID offset buffer and the dynamic offset buffer to use for chunking
// up a large compute dispatch. The start of the push constants data will be:
// {
//      u32: global work group id offset
//      u32: totalWorkGroups
//      ...: up to 248 bytes additional data (if any) from the pushConstants parameter,
//           passed as an ArrayBuffer
// }
// ID offset (u32),
function buildPushConstantsBuffer(device, totalWorkGroups, pushConstants) {
  var dynamicOffsets = [];
  var dispatchSizes = [];

  //maxComputeWorkgroupsPerDimension is maximum value for number of workgroup.xyz. (current state: 65535)
  var numDynamicOffsets = Math.ceil(
    totalWorkGroups / device.limits.maxComputeWorkgroupsPerDimension
  );

  var idOffsetsBuffer = device.createBuffer({
    size: 256 * numDynamicOffsets,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  {
    var pushConstantsView = null;
    if (pushConstants) {
      pushConstantsView = new Uint8Array(pushConstants);
      console.log(`got push constants of size ${pushConstantsView.length}`);
    }
    var mapping = idOffsetsBuffer.getMappedRange();
    for (var i = 0; i < numDynamicOffsets; ++i) {
      dynamicOffsets.push(i * 256);

      if (i + 1 < numDynamicOffsets) {
        dispatchSizes.push(device.limits.maxComputeWorkgroupsPerDimension);
      } else {
        dispatchSizes.push(
          totalWorkGroups % device.limits.maxComputeWorkgroupsPerDimension
        );
      }

      // Write the push constants data
      // last argument which is length is not length in byte but number of element of ui32
      var u32view = new Uint32Array(mapping, i * 256, 2);
      u32view[0] = device.limits.maxComputeWorkgroupsPerDimension * i;
      u32view[1] = totalWorkGroups;

      // Copy in any additional push constants data if provided
      if (pushConstantsView) {
        var u8view = new Uint8Array(mapping, i * 256 + 8, 248);
        u8view.set(pushConstantsView);
      }
    }
    idOffsetsBuffer.unmap();
  }
  dynamicOffsets = new Uint32Array(dynamicOffsets);

  return {
    nOffsets: numDynamicOffsets,
    gpuBuffer: idOffsetsBuffer,
    dynamicOffsets: dynamicOffsets,
    dispatchSizes: dispatchSizes,
  };
}
