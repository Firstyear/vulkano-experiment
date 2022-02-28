// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! This example contains the source code of the second part of the guide at http://vulkano.rs.
//!
//! It is not commented, as the explanations can be found in the guide itself.

use std::sync::Arc;
use std::time::Instant;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::instance::{Instance, InstanceExtensions};
use vulkano::pipeline::Pipeline;
use vulkano::pipeline::{ComputePipeline, PipelineBindPoint};
use vulkano::sync;
use vulkano::sync::GpuFuture;
use vulkano::Version;


fn main() {
    let instance = Instance::new(None, Version::V1_1, &InstanceExtensions::none(), None)
        .expect("failed to create instance");

    PhysicalDevice::enumerate(&instance).for_each(|phy| {
        let props = phy.properties();
        println!(
            "Device -> {}, ACU -> {:?}, DID -> {}, MBS -> {:?}",
            props.device_name,
            props.active_compute_unit_count,
            props.device_id,
            props.max_buffer_size,
        );
    });

    let physical = PhysicalDevice::enumerate(&instance)
        .next()
        .expect("no device available");

    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_compute())
        .expect("couldn't find a compute queue family");

    let (device, mut queues) = {
        Device::new(
            physical,
            &Features::none(),
            &DeviceExtensions {
                khr_storage_buffer_storage_class: true,
                khr_portability_subset: true,
                ..DeviceExtensions::none()
            },
            [(queue_family, 0.5)].iter().cloned(),
        )
        .expect("failed to create device")
    };

    println!("{} queue(s) available", queues.len());
    let queue = queues.next().unwrap();

    let data_iter = 0..65536u32;

    let immut_buffer = {
        let (immut_buffer, immut_buffer_future) =
            ImmutableBuffer::from_iter(data_iter, BufferUsage::all(), queue.clone())
                .expect("failed to create buffer");

        let mut builder = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        immut_buffer_future
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        println!("Immuted buffer is ready!");

        immut_buffer
    };

    let dev_buffer: Arc<DeviceLocalBuffer<[u32]>> = DeviceLocalBuffer::array(
        device.clone(),
        65536,
        BufferUsage::transfer_source()
            | BufferUsage::transfer_destination()
            | BufferUsage::storage_buffer(),
        Some(queue.family()),
    )
    .expect("failed to create buffer");

    let res_iter = (0..65536).map(|_| 0u32);
    let res_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::transfer_destination(),
        true,
        res_iter,
    )
    .expect("failed to create buffer");

    // Compute pipelines

    /* NOTES!!!! */
    /*
     * The local size is 64 per work group. Later on in dispatch we spawn 1024 work groups. This
     * yields 1024 * local_size == 65536.
     *
     * Docs state that local_size should always be 32 or 64. Then adjust work groups accordingly!
     */

    let pipeline = {
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                src: "
    #version 450

    layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

    layout(set = 0, binding = 0) buffer Data {
        uint data[];
    } buf;

    layout(set = 0, binding = 1) readonly restrict buffer ImmutableData {
        uint data[];
    } immutable_data;

    void main() {
        uint idx = gl_GlobalInvocationID.x;
        buf.data[idx] = immutable_data.data[idx] + 1;
    }"
            }
        }

        let shader = cs::load(device.clone()).expect("failed to create shader module");
        ComputePipeline::new(
            device.clone(),
            shader.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .expect("failed to create compute pipeline")
    };

    let start = Instant::now();

    let layout = pipeline.layout().descriptor_set_layouts().get(0).unwrap();

    let set = PersistentDescriptorSet::new(
        layout.clone(),
        [
            WriteDescriptorSet::buffer(0, dev_buffer.clone()), // 0 is the binding
            WriteDescriptorSet::buffer(1, immut_buffer.clone()), // 0 is the binding
        ],
    )
    .unwrap();

    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.family(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    // builder.copy_buffer(immut_buffer.clone(), dev_buffer.clone()).unwrap();

    builder
        .bind_pipeline_compute(pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0, // 0 is the index of our set
            set,
        )
        .dispatch([1024, 1, 1])
        .unwrap();

    builder
        .copy_buffer(dev_buffer.clone(), res_buffer.clone())
        .unwrap();

    let command_buffer = builder.build().unwrap();

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    println!("Took - {:?}", start.elapsed());

    let content = res_buffer.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, (n as u32) + 1);
    }

    let a = Vec::from_iter(0..65536u32);
    let mut b = Vec::from_iter((0..65536).map(|_| 0u32));

    let start = Instant::now();

    for i in 0..65536 {
        b[i] = a[i] + 1;
    }

    println!("Took - {:?}", start.elapsed());

    println!("Everything succeeded!");
}
