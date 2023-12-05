import gradio as gr

str_input_code = """class GraphFuseDemo(torch.nn.Module):
    def forward(
        self,
        a0: torch.tensor,
        a1: torch.tensor,
        e0: torch.tensor,
    ):
        b0 = a0 * a1
        b_out0 = b0 * 2.0
        c_out0 = b_out0 - 1.42
        d_out0 = torch.sigmoid(b_out0)
        e_out0 = torch.relu(e0)
        return c_out0, d_out0, e_out0
"""

str_output_code = """ @triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 4096)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp8 = tl.load(in_ptr2 + (x2), xmask)
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp5 = 1.42
    tmp6 = tmp4 - tmp5
    tmp7 = tl.sigmoid(tmp4)
    tmp9 = tl.where(0 != 0, 0, tl.where(0 > tmp8, 0, tmp8))
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp6, xmask)
    tl.store(out_ptr1 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp7, xmask)
    tl.store(out_ptr2 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp9, xmask)
"""

with gr.Blocks() as demo:
    gr.Markdown("Welder online demo to support triton compile and optimization.")
    with gr.Tab("Welder"):
        with gr.Row():
            with gr.Column():
                example_key = gr.Dropdown(
                    ["(Null)", "MatMul", "Convolution", "Elementwise"],
                    value="(Null)",
                    label="Kernel",
                    info="Preset Kernel Example",
                )
                hardware_platform = gr.Radio(
                    ["Standard_NC24ads_A100_v4", "AMD MI250", "MSFT MIAI100"],
                    value="Standard_NC24ads_A100_v4",
                    label="Hardware",
                    info="Hardware Platform",
                )
                input_type = gr.Radio(
                    ["Pytorch nn.Module", "Einstain v2 Expression"],
                    label="Input Code Type",
                    value="Pytorch nn.Module",
                )
                input_code = (
                    gr.Code(
                        value=str_input_code, label="Code", lines=25, language="python"
                    ),
                )
            with gr.Column():
                with gr.Row():
                    compile_button = (gr.Button(value="Compile"),)
                    copy_button = (gr.Button(value="Share"),)
                output_code = (
                    gr.Code(
                        value=str_output_code,
                        label="Output",
                        lines=15,
                        language="python",
                    ),
                )
                with gr.Tab("Trace"):
                    profile_graph = gr.Plot(label="Profile")
                with gr.Tab("Log"):
                    pass

    with gr.Tab("Changelog"):
        pass


if __name__ == "__main__":
    demo.launch(show_api=False)
