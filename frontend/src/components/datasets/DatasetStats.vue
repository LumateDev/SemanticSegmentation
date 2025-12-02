<template>
  <el-dialog v-model="visible" :title="`Статистика: ${fileName}`" width="700px" destroy-on-close>
    <!-- Loading -->
    <div v-if="isLoadingStats" v-loading="true" class="dataset-stats__loading" />

    <!-- Error -->
    <el-alert v-else-if="statsError" :title="statsError" type="error" show-icon />

    <!-- Content -->
    <div v-else-if="currentStats" class="dataset-stats">
      <!-- Summary -->
      <el-row :gutter="16" class="dataset-stats__summary">
        <el-col :span="8">
          <div class="dataset-stats__stat-item">
            <span class="dataset-stats__stat-label">Всего точек</span>
            <span class="dataset-stats__stat-value">
              {{ formatNumber(currentStats.total_points) }}
            </span>
          </div>
        </el-col>
        <el-col :span="8">
          <div class="dataset-stats__stat-item">
            <span class="dataset-stats__stat-label">Классов</span>
            <span class="dataset-stats__stat-value">
              {{ currentStats.unique_classes }}
            </span>
          </div>
        </el-col>
        <el-col :span="8">
          <div class="dataset-stats__stat-item">
            <span class="dataset-stats__stat-label">Файл</span>
            <span class="dataset-stats__stat-value dataset-stats__stat-value--small">
              {{ currentStats.file }}
            </span>
          </div>
        </el-col>
      </el-row>

      <!-- Chart -->
      <div class="dataset-stats__chart">
        <h4 class="dataset-stats__title">Распределение по классам</h4>
        <div ref="chartRef" class="dataset-stats__chart-container" />
      </div>

      <!-- Table -->
      <div class="dataset-stats__table">
        <h4 class="dataset-stats__title">Детализация</h4>
        <el-table :data="classStatsArray" stripe size="small">
          <el-table-column prop="name" label="Класс" min-width="150">
            <template #default="{ row }">
              <div class="dataset-stats__class-cell">
                <span class="dataset-stats__class-dot" :style="{ backgroundColor: getClassColor(row.label) }" />
                <span>{{ row.name }}</span>
              </div>
            </template>
          </el-table-column>
          <el-table-column prop="label" label="Метка" width="80" align="center" />
          <el-table-column prop="count" label="Количество" width="120" align="right">
            <template #default="{ row }">
              {{ formatNumber(row.count) }}
            </template>
          </el-table-column>
          <el-table-column prop="percentage" label="Доля" width="180">
            <template #default="{ row }">
              <div class="dataset-stats__progress-cell">
                <el-progress
                  :percentage="row.percentage"
                  :stroke-width="8"
                  :show-text="false"
                  :color="getClassColor(row.label)"
                />
                <span>{{ row.percentage.toFixed(2) }}%</span>
              </div>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </div>

    <!-- Empty -->
    <el-empty v-else description="Нет данных" />

    <template #footer>
      <el-button @click="visible = false">Закрыть</el-button>
      <el-button type="primary" :icon="Download" @click="handleExport"> Экспорт JSON </el-button>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick, onUnmounted } from 'vue';
import { storeToRefs } from 'pinia';
import { Download } from '@element-plus/icons-vue';
import { useDatasetsStore } from '@/stores/datasetsStore';
import * as echarts from 'echarts';

const props = defineProps<{
  modelValue: boolean;
  fileName: string;
  subdir?: string;
}>();

const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void;
}>();

const datasetsStore = useDatasetsStore();
const { currentStats, isLoadingStats, statsError, classStatsArray } = storeToRefs(datasetsStore);

const chartRef = ref<HTMLElement | null>(null);
let chartInstance: echarts.ECharts | null = null;

const visible = computed({
  get: () => props.modelValue,
  set: value => emit('update:modelValue', value),
});

const CLASS_COLORS: Record<number, string> = {
  0: '#909399',
  2: '#8B4513',
  3: '#228B22',
  4: '#32CD32',
  5: '#006400',
  6: '#FF6347',
  9: '#4169E1',
};

function getClassColor(label: number): string {
  return CLASS_COLORS[label] || '#409EFF';
}

function formatNumber(num: number): string {
  return num.toLocaleString('ru-RU');
}

function initChart(): void {
  if (!chartRef.value || !classStatsArray.value.length) return;

  chartInstance = echarts.init(chartRef.value);

  const option: echarts.EChartsOption = {
    tooltip: {
      trigger: 'item',
      formatter: '{b}: {c} ({d}%)',
    },
    legend: {
      orient: 'vertical',
      right: 10,
      top: 'center',
    },
    series: [
      {
        type: 'pie',
        radius: ['40%', '70%'],
        center: ['40%', '50%'],
        itemStyle: {
          borderRadius: 4,
          borderColor: '#fff',
          borderWidth: 2,
        },
        label: { show: false },
        emphasis: {
          label: {
            show: true,
            fontSize: 14,
            fontWeight: 'bold',
          },
        },
        data: classStatsArray.value.map(item => ({
          value: item.count,
          name: item.name,
          itemStyle: { color: getClassColor(item.label) },
        })),
      },
    ],
  };

  chartInstance.setOption(option);
}

function handleExport(): void {
  if (!currentStats.value) return;

  const data = {
    ...currentStats.value,
    exported_at: new Date().toISOString(),
  };

  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `stats_${props.fileName}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

watch(visible, async newVal => {
  if (newVal && props.fileName) {
    await datasetsStore.fetchStats(props.fileName, props.subdir || 'raw');
  } else {
    datasetsStore.clearStats();
    chartInstance?.dispose();
    chartInstance = null;
  }
});

watch(currentStats, async () => {
  if (currentStats.value) {
    await nextTick();
    initChart();
  }
});

onUnmounted(() => {
  chartInstance?.dispose();
});
</script>

<style lang="scss" scoped>
.dataset-stats {
  &__loading {
    min-height: 200px;
  }

  &__summary {
    margin-bottom: 24px;
    padding: 20px;
    background: var(--el-fill-color-lighter);
    border-radius: 8px;
  }

  &__stat-item {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  &__stat-label {
    font-size: 13px;
    color: var(--el-text-color-secondary);
  }

  &__stat-value {
    font-size: 24px;
    font-weight: 600;
    color: var(--el-color-primary);
    line-height: 1.2;

    &--small {
      font-size: 14px;
      font-weight: 500;
      word-break: break-all;
      color: var(--el-text-color-regular);
    }
  }

  &__title {
    margin: 0 0 12px;
    font-size: 14px;
    font-weight: 500;
  }

  &__chart {
    margin-bottom: 24px;
  }

  &__chart-container {
    width: 100%;
    height: 280px;
  }

  &__class-cell {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  &__class-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  &__progress-cell {
    display: flex;
    align-items: center;
    gap: 8px;

    .el-progress {
      flex: 1;
    }

    span {
      min-width: 55px;
      text-align: right;
      font-size: 12px;
      color: var(--el-text-color-secondary);
    }
  }
}
</style>
