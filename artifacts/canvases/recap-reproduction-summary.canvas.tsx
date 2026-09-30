import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Code,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  LineChart,
  Pill,
  Row,
  Spacer,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
} from "cursor/canvas";

const valueCkpts = [
  { step: 2520, loss: 2.79, mae: 0.107, spearman: 0.647 },
  { step: 2640, loss: 2.91, mae: 0.104, spearman: 0.743 },
  { step: 2760, loss: 2.94, mae: 0.104, spearman: 0.685 },
  { step: 2880, loss: 2.83, mae: 0.0903, spearman: 0.738 },
  { step: 3000, loss: 2.75, mae: 0.0957, spearman: 0.76 },
];

const cfgCkpts = [
  {
    step: "3000",
    loss: 0.0126,
    grad: 0.124,
    cond: 0.281,
    pos: 0.312,
  },
  {
    step: "6000",
    loss: 0.0117,
    grad: 0.103,
    cond: 0.273,
    pos: 0.305,
  },
  {
    step: "9000",
    loss: 0.0125,
    grad: 0.105,
    cond: 0.352,
    pos: 0.383,
  },
];

const evalTask0 = [
  { label: "9000 smoke×3", rate: 33 },
  { label: "9000 Task0×10", rate: 30 },
  { label: "6000 Task0×10", rate: 40 },
];

const suite9000 = [
  { task: "T0 soup+sauce→basket", rate: 30 },
  { task: "T1 cream+butter", rate: 0 },
  { task: "T2 stove+moka", rate: 0 },
  { task: "T3 bowl+drawer", rate: 0 },
  { task: "T4 mug+plate", rate: 0 },
  { task: "T5 book", rate: 0 },
  { task: "T6 mug+pudding", rate: 0 },
  { task: "T7 soup+cream", rate: 0 },
  { task: "T8 dual moka", rate: 0 },
  { task: "T9 mug+microwave", rate: 0 },
];

export default function RecapReproductionSummary() {
  const { tokens: t } = useHostTheme();

  return (
    <Stack gap={24} style={{ maxWidth: 980 }}>
      <Stack gap={8}>
        <H1>RLinf RECAP 复现总览</H1>
        <Text tone="secondary">
          Libero Task0 离线四步管线 → CFG 训至 9000 停 → 仿真 vanilla
          评测 → 6000/9000 对照。范围是 pipeline 跑通 + Task0
          证据，不是论文量级全量 CFG-guidance 评测。
        </Text>
        <Row gap={8} style={{ flexWrap: "wrap" }}>
          <Pill tone="success" active>
            Step1–4 完成
          </Pill>
          <Pill tone="warning" active>
            CFG 停于 9000 / 30000
          </Pill>
          <Pill tone="info" active>
            评测 = vanilla（无 guidance）
          </Pill>
          <Pill>Task0 域内 ~30–40%</Pill>
        </Row>
      </Stack>

      <Grid columns={4} gap={12}>
        <Stat value="4/4" label="管线步骤" tone="success" />
        <Stat value="9000" label="CFG 最终步" />
        <Stat value="40%" label="6000 Task0×10" tone="success" />
        <Stat value="30%" label="9000 Task0×10" />
      </Grid>

      <Callout tone="info" title="一句话结论">
        离线 RECAP（returns→value→advantages→CFG）在官方 Task0
        数据上已端到端跑通；中途权重可导出并在 Libero 仿真里成功。当前数字证明的是「管线与权重可用 +
        Task0 有非零成功率」，不能证明「CFG guidance
        带来增益」或「训满 30k 后的最终性能」。
      </Callout>

      <Divider />

      <H2>算法骨架（关键点）</H2>
      <Grid columns={2} gap={12}>
        <Card>
          <CardHeader>RECAP 在做什么</CardHeader>
          <CardBody>
            <Stack gap={6}>
              <Text>
                用轨迹回报监督 value：V̂(s) ≈ R_t（此处
                fail300：失败帧 reward=-300）。
              </Text>
              <Text>
                用 V̂ 标 advantage，再条件化策略（RLinf 实现为 CFG：正/负条件 +
                guidance scale）。
              </Text>
              <Text weight="semibold">
                注意：A ≠ Q−V 的在线 TD；本管线是离线 return→value→advantage→CFG。
              </Text>
            </Stack>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>本次刻意收敛的范围</CardHeader>
          <CardBody>
            <Stack gap={6}>
              <Text>数据：官方 RECAP-Libero10-Task0-48succ（~82G）</Text>
              <Text>底模：π0.5 PyTorch + SigLIP2 + Gemma3-270M</Text>
              <Text>CFG 计划 30000，磁盘/时间约束下停在 9000，保留 6000</Text>
              <Text>
                仿真评测用 standalone OpenPI loader = vanilla
                前向，未接 cfgrl_guidance_scale
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Divider />

      <H2>四步管线：做了什么 / 结果说明什么</H2>

      <Card>
        <CardHeader trailing={<Pill tone="success" size="sm" active>完成</Pill>}>
          Step1 · Compute Returns（tag=fail300）
        </CardHeader>
        <CardBody>
          <Stack gap={8}>
            <Text>
              对 SFT + rollout 轨迹按 γ=1.0 回填 return-to-go；失败帧
              failure_reward=-300，产出后续 value 监督目标。
            </Text>
            <Text tone="secondary">
              说明：回报分布决定 value 学什么；fail300
              把失败拉得很负，advantage 边界会更清晰。
            </Text>
            <Text size="small" tone="secondary">
              图：{" "}
              <Code>
                /root/rlinf_recap/logs/plots/01_return_to_go_value_targets.png
              </Code>
            </Text>
          </Stack>
        </CardBody>
      </Card>

      <Card>
        <CardHeader trailing={<Pill tone="success" size="sm" active>完成</Pill>}>
          Step2 · Value Model SFT → global_step_3000
        </CardHeader>
        <CardBody>
          <Stack gap={12}>
            <Text>
              SigLIP2+Gemma critic 回归 R_t。中途 ckpt 训练日志：MAE≈0.09–0.11，Spearman
              从 0.65→0.76（@3000）。无独立 val / 在线 rollout。
            </Text>
            <LineChart
              categories={valueCkpts.map((d) => String(d.step))}
              series={[
                {
                  name: "Spearman ρ",
                  data: valueCkpts.map((d) => d.spearman),
                  tone: "info",
                },
                {
                  name: "MAE",
                  data: valueCkpts.map((d) => d.mae),
                  tone: "warning",
                },
              ]}
              height={200}
            />
            <Text size="small" tone="secondary">
              Source: value SFT train log @ mid-ckpts · Val disabled
            </Text>
            <Text size="small" tone="secondary">
              图集：{" "}
              <Code>/root/rlinf_recap/logs/plots_value_midckpts/</Code> ·{" "}
              <Code>01_value_sft_loss_mae_midckpts.png</Code> ·{" "}
              <Code>02_value_sft_spearman_catacc.png</Code>
            </Text>
            <Text>
              说明：value 对 return
              排序已可用，作为 advantage 标注器够格；不等于仿真成功率。
            </Text>
          </Stack>
        </CardBody>
      </Card>

      <Card>
        <CardHeader trailing={<Pill tone="success" size="sm" active>完成</Pill>}>
          Step3 · Compute Advantages
        </CardHeader>
        <CardBody>
          <Stack gap={8}>
            <Text>
              加载 value@3000，对轨迹帧写 advantage / 正负条件标签，供 CFG
              条件化采样。
            </Text>
            <Text tone="secondary">
              说明：这一步是「把价值判断写进数据」；之后策略学的是条件分布，不是再训
              V。
            </Text>
            <Text size="small" tone="secondary">
              优势分布图：{" "}
              <Code>
                /root/rlinf_recap/logs/plots_cfg3000/05_advantage_dist_for_cfg.png
              </Code>
            </Text>
          </Stack>
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          trailing={
            <Pill tone="warning" size="sm" active>
              停于 9000
            </Pill>
          }
        >
          Step4 · CFG RL on π0.5（目标 30000）
        </CardHeader>
        <CardBody>
          <Stack gap={12}>
            <Text>
              OpenPI CFG SFT：loss 从 ~0.073(@50) 降到 ~0.012 量级并平台化；3000 /
              6000 / 9000 的 loss≈0.0117–0.0126。条件占比
              cond/pos 在 9000 略升（cond 0.35，pos 0.38）。
            </Text>
            <Grid columns={2} gap={12}>
              <BarChart
                categories={cfgCkpts.map((d) => d.step)}
                series={[
                  {
                    name: "CFG loss",
                    data: cfgCkpts.map((d) => d.loss),
                    tone: "info",
                  },
                ]}
                height={180}
              />
              <BarChart
                categories={cfgCkpts.map((d) => d.step)}
                series={[
                  {
                    name: "cond ratio",
                    data: cfgCkpts.map((d) => d.cond),
                    tone: "success",
                  },
                  {
                    name: "pos ratio",
                    data: cfgCkpts.map((d) => d.pos),
                    tone: "warning",
                  },
                ]}
                height={180}
              />
            </Grid>
            <Text size="small" tone="secondary">
              Source: CFG train metrics @ mid-ckpts · planned 30k, stopped 9k
            </Text>
            <Text size="small" tone="secondary">
              对比图集：{" "}
              <Code>/root/rlinf_recap/logs/plots_cfg_compare_3ckpts/</Code> ·
              单点：
              <Code>plots_cfg6000/</Code>、<Code>plots_cfg9000/</Code>
            </Text>
            <Text>
              说明：训练曲线健康、可收敛；loss
              平台不代表策略已最优。停训是工程决策（磁盘/时间），不是算法收敛判据。
            </Text>
          </Stack>
        </CardBody>
      </Card>

      <Divider />

      <H2>仿真评测：从导出到对照</H2>
      <Text tone="secondary">
        导出：RLinf <Code>full_weights.pt</Code> → deploy{" "}
        <Code>model.safetensors</Code> +{" "}
        <Code>physical-intelligence/libero/norm_stats.json</Code>。渲染
        osmesa；禁用 torch.compile。
      </Text>

      <Grid columns={2} gap={12}>
        <Card>
          <CardHeader>Task0 成功率对照</CardHeader>
          <CardBody>
            <BarChart
              categories={evalTask0.map((d) => d.label)}
              series={[
                {
                  name: "success %",
                  data: evalTask0.map((d) => d.rate),
                  tone: "success",
                },
              ]}
              height={200}
            />
            <Text size="small" tone="secondary">
              Source: summary.json · vanilla pi05 · no CFG guidance
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>9000 · libero_10×10 逐任务</CardHeader>
          <CardBody>
            <BarChart
              categories={suite9000.map((d) => d.task.slice(0, 8))}
              series={[
                {
                  name: "success %",
                  data: suite9000.map((d) => d.rate),
                  tone: "neutral",
                },
              ]}
              height={200}
            />
            <Text size="small" tone="secondary">
              Overall 3%（3/100）· ~3.4h · Task1–9 全 0
            </Text>
          </CardBody>
        </Card>
      </Grid>

      <Table
        headers={["评测", "设置", "结果", "说明"]}
        rows={[
          [
            "9000 smoke",
            "Task0 × 3",
            "33% (1/3)",
            "通路冒烟：能加载、能交互、有成功",
          ],
          [
            "9000 加长",
            "libero_10 × 10",
            "overall 3%；Task0 30%",
            "域内 Task0 非零；其余任务 OOD→0",
          ],
          [
            "6000 对照",
            "Task0 × 10",
            "40% (4/10)",
            "与 9000 同档；差 1 局属噪声",
          ],
        ]}
        rowTone={[undefined, "success", "info"]}
      />

      <Callout tone="warning" title="评测口径限制（读数前必看）">
        standalone OpenPI loader 走的是普通 π0.5
        前向，没有 dual-forward CFG guidance。因此：6000/9000
        对比的是「中途权重当普通策略」的表现，不是论文里
        cfgrl_guidance_scale 条件下的 RECAP 正式分。
      </Callout>

      <Divider />

      <H2>做到哪一步、能分析出什么</H2>
      <Grid columns={2} gap={12}>
        <Card>
          <CardHeader>已经能成立的主张</CardHeader>
          <CardBody>
            <Stack gap={6}>
              <Text>RLinf RECAP 四步脚本/配置/数据链路在本机可复现</Text>
              <Text>Value@3000 对 return 有可用排序相关（ρ≈0.76）</Text>
              <Text>CFG 训练稳定，9000 步可导出并在 Libero 仿真 rollout</Text>
              <Text>
                Task0（训练域）vanilla 成功率约 30–40%；非 Task0 任务接近 0
              </Text>
              <Text>6000 vs 9000 无显著优劣（小样本）</Text>
            </Stack>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>还不能下的结论</CardHeader>
          <CardBody>
            <Stack gap={6}>
              <Text>CFG guidance 是否优于 vanilla / 优于 base π0.5</Text>
              <Text>训满 30k 是否继续涨分</Text>
              <Text>跨任务泛化（suite 其余任务未专项训）</Text>
              <Text>与真机 / pistar JAX RECAP 的数值对齐</Text>
              <Text>Value 中途 ckpt 谁做标注器更优（只用了 3000）</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <H3>建议的下一刀（若要升级证据等级）</H3>
      <Table
        headers={["优先级", "动作", "能补上的缺口"]}
        rows={[
          [
            "P0",
            "接 CFG guidance 评测入口",
            "区分「权重可用」vs「RECAP 条件推理有效」",
          ],
          [
            "P1",
            "同一设置跑 base π0.5 Task0×10/×50",
            "是否相对底模有增益",
          ],
          [
            "P2",
            "续训或至少再留 12k/15k ckpt",
            "看中后期是否过拟合 / 继续涨",
          ],
          [
            "P3",
            "Task0×50 扩样本",
            "压低 6000/9000 对比方差",
          ],
        ]}
      />

      <Text size="small" tone="secondary">
        产物根目录：/root/rlinf_recap · 数据
        datasets/RECAP-Libero10-Task0-48succ-Data · CFG ckpt
        RLinf/logs/cfg_rl/.../global_step_&#123;6000,9000&#125; · deploy
        deploy/cfg_rl_step&#123;6000,9000&#125; · 评测日志
        logs/libero_eval_cfg*
      </Text>
    </Stack>
  );
}
