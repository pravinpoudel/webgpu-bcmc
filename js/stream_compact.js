var StreamCompact = function(device) {
    this.device = device;

    // Not sure how to query this limit, assuming this size based on OpenGL
    // In a less naive implementation doing some block-based implementation w/
    // larger group sizes might be better as well
    // We also need to make sure the offset we'll end up using for the
    // dynamic offsets is aligned to 256 bytes. We're offsetting into arrays
    // of uint32, so determine the max dispatch size we should use for each
    // individual aligned chunk
    this.maxDispatchSize = Math.floor((2 * 65535 * 4) / 256) * 256;

    this.streamCompactBGLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage",
                }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage",
                }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "uniform",
                }
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage",
                }
            },
        ],
    });
    this.streamCompactPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [this.streamCompactBGLayout],
        }),
        compute: {
            module: device.createShaderModule({code: stream_compact_comp_spv}),
            entryPoint: "main",
        },
    });

    this.dataBGLayout = device.createBindGroupLayout({
        entries: [{
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: "storage",
            }
        }]
    });
    this.streamCompactDataPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout(
            {bindGroupLayouts: [this.streamCompactBGLayout, this.dataBGLayout]}),
        compute: {
            module: device.createShaderModule({code: stream_compact_data_comp_spv}),
            entryPoint: "main",
        }
    });
};

StreamCompact.prototype.compactActiveIDs =
    async function(numElements, isActiveBuffer, offsetsBuffer, outputBuffer) {
    // No push constants in the API? This is really a hassle to hack together
    // because I also have to obey (at least Dawn's rule is it part of the spec?)
    // that the dynamic offsets be 256b aligned
    // Please add push constants!
    var numChunks = Math.ceil(numElements / this.maxDispatchSize);
    var compactPassOffset = this.device.createBuffer({
        size: numChunks * 256,
        usage: GPUBufferUsage.UNIFORM,
        mappedAtCreation: true,
    });
    {
        var map = new Uint32Array(compactPassOffset.getMappedRange());
        for (var i = 0; i < numChunks; ++i) {
            map[i * 64] = i * this.maxDispatchSize;
        }
        compactPassOffset.unmap();
    }
    var commandEncoder = this.device.createCommandEncoder();
    var pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.streamCompactPipeline);
    for (var i = 0; i < numChunks; ++i) {
        var numWorkGroups =
            Math.min(numElements - i * this.maxDispatchSize, this.maxDispatchSize);
        var offset = i * this.maxDispatchSize * 4;
        // Have to create bind groups here because dynamic offsets are not allowed
        var streamCompactBG = null;
        if (numWorkGroups === this.maxDispatchSize) {
            // TODO: Probably doesn't need to be implemented this way with the offsets
            // for the isactive and offsets buffers
            streamCompactBG = this.device.createBindGroup({
                layout: this.streamCompactBGLayout,
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: isActiveBuffer,
                            size: 4 * Math.min(numElements, this.maxDispatchSize),
                            offset: offset,
                        },
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: offsetsBuffer,
                            size: 4 * Math.min(numElements, this.maxDispatchSize),
                            offset: offset,
                        },
                    },
                    {
                        binding: 2,
                        resource: {
                            buffer: compactPassOffset,
                            size: 4,
                            offset: i * 256,
                        },
                    },
                    {
                        binding: 3,
                        resource: {
                            buffer: outputBuffer,
                        },
                    },
                ],
            });
        } else {
            streamCompactBG = this.device.createBindGroup({
                layout: this.streamCompactBGLayout,
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: isActiveBuffer,
                            size: 4 * (numElements % this.maxDispatchSize),
                            offset: offset,
                        },
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: offsetsBuffer,
                            size: 4 * (numElements % this.maxDispatchSize),
                            offset: offset,
                        },
                    },
                    {
                        binding: 2,
                        resource: {
                            buffer: compactPassOffset,
                            size: 4,
                            offset: i * 256,
                        },
                    },
                    {
                        binding: 3,
                        resource: {
                            buffer: outputBuffer,
                        },
                    },
                ],
            });
        }
        pass.setBindGroup(0, streamCompactBG);
        pass.dispatch(numWorkGroups, 1, 1);
    }
    pass.endPass();
    this.device.queue.submit([commandEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
};

StreamCompact.prototype.compactActive =
    async function(numElements, isActiveBuffer, offsetsBuffer, dataBuffer, outputBuffer) {
    // No push constants in the API? This is really a hassle to hack together
    // because I also have to obey (at least Dawn's rule is it part of the spec?)
    // that the dynamic offsets be 256b aligned
    // Please add push constants!
    var numChunks = Math.ceil(numElements / this.maxDispatchSize);
    var compactPassOffset = this.device.createBuffer({
        size: numChunks * 256,
        usage: GPUBufferUsage.UNIFORM,
        mappedAtCreation: true,
    });
    {
        var map = new Uint32Array(compactPassOffset.getMappedRange());
        for (var i = 0; i < numChunks; ++i) {
            map[i * 64] = i * this.maxDispatchSize;
        }
        compactPassOffset.unmap();
    }

    var dataBG = this.device.createBindGroup(
        {layout: this.dataBGLayout, entries: [{binding: 0, resource: {buffer: dataBuffer}}]});

    var commandEncoder = this.device.createCommandEncoder();
    var pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.streamCompactDataPipeline);
    for (var i = 0; i < numChunks; ++i) {
        var numWorkGroups =
            Math.min(numElements - i * this.maxDispatchSize, this.maxDispatchSize);

        // Have to create bind groups here because dynamic offsets are not allowed
        var streamCompactBG = this.device.createBindGroup({
            layout: this.streamCompactBGLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: isActiveBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: offsetsBuffer,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: compactPassOffset,
                        size: 4,
                        offset: i * 256,
                    },
                },
                {
                    binding: 3,
                    resource: {
                        buffer: outputBuffer,
                    },
                },
            ],
        });
        pass.setBindGroup(0, streamCompactBG);
        pass.setBindGroup(1, dataBG);
        pass.dispatch(numWorkGroups, 1, 1);
    }
    pass.endPass();
    this.device.queue.submit([commandEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
};
