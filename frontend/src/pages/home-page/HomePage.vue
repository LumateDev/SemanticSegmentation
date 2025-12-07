<template>
  <div class="home-page">
    <el-card class="home-page__card" shadow="hover">
      <template #header>
        <div class="home-page__header">
          <el-icon size="24" class="home-page__icon">
            <Cpu />
          </el-icon>
          <h2 class="home-page__title">Программное решение для распознавания местности на основе нейронных сетей</h2>
        </div>
      </template>

      <div class="home-page__content">
        <!-- Основные возможности -->
        <el-row :gutter="20" class="home-page__features">
          <el-col :xs="24" :sm="12" class="home-page__feature-col">
            <el-card shadow="never" class="home-page__feature-card">
              <div class="home-page__feature-icon">
                <el-icon size="32" color="var(--el-color-primary)">
                  <DataAnalysis />
                </el-icon>
              </div>
              <h3 class="home-page__feature-title">Обучение моделей</h3>
              <p class="home-page__feature-desc">
                Тренируйте нейронные сети DGCNN на данных местности. Настройте параметры обучения и запускайте процесс.
              </p>
              <el-button type="primary" @click="$router.push('/train')">
                <el-icon><VideoPlay /></el-icon>
                Начать обучение
              </el-button>
            </el-card>
          </el-col>

          <el-col :xs="24" :sm="12" class="home-page__feature-col">
            <el-card shadow="never" class="home-page__feature-card">
              <div class="home-page__feature-icon">
                <el-icon size="32" color="var(--el-color-success)">
                  <FolderOpened />
                </el-icon>
              </div>
              <h3 class="home-page__feature-title">Управление данными</h3>
              <p class="home-page__feature-desc">
                Загружайте и управляйте .xyz датасетами, просматривайте существующие модели и результаты обучения.
              </p>
              <el-button type="success" @click="$router.push('/datasets')">
                <el-icon><Folder /></el-icon>
                Управление данными
              </el-button>
            </el-card>
          </el-col>
        </el-row>

        <!-- Процесс работы -->
        <el-card class="home-page__process" shadow="never">
          <template #header>
            <div class="home-page__process-header">
              <el-icon><Operation /></el-icon>
              <span>Процесс работы</span>
            </div>
          </template>
          <div class="home-page__process-steps">
            <div class="home-page__process-step">
              <div class="home-page__step-number">1</div>
              <div class="home-page__step-content">
                <h4>Подготовка данных</h4>
                <p>Загрузите .xyz файлы с данными местности в разделе "Управление данными"</p>
              </div>
            </div>
            <div class="home-page__process-step">
              <div class="home-page__step-number">2</div>
              <div class="home-page__step-content">
                <h4>Конфигурация обучения</h4>
                <p>Настройте параметры модели: batch size, learning rate, количество эпох</p>
              </div>
            </div>
            <div class="home-page__process-step">
              <div class="home-page__step-number">3</div>
              <div class="home-page__step-content">
                <h4>Запуск обучения</h4>
                <p>Запустите процесс обучения и отслеживайте прогресс в реальном времени</p>
              </div>
            </div>
          </div>
        </el-card>

        <!-- Информация о системе -->
        <el-alert
          title="Технологии"
          type="info"
          :closable="false"
          class="home-page__tech-info"
        >
          <div class="home-page__tech-scroll">
            <p>• Бэкенд: Python</p>
            <p>• Фронтенд: Vue js, Element Plus</p>
            <p>• Нейросеть: DGCNN (Dynamic Graph CNN)</p>
            <p>• Данные: Трехмерные облака точек в формате .xyz</p>
            <p>• Коммуникация: WebSocket для реального времени</p>
            <p>• API: REST + WebSocket endpoints</p>
            <p>• Форматы данных: JSON для конфигурации</p>
            <p>• Логирование: Реал-тайм логи через WebSocket</p>
          </div>
        </el-alert>

        <!-- Подпись -->
        <div class="home-page__footer">
          <div class="home-page__footer-content">
            <el-divider />
            <div class="home-page__group-info">
              <p class="home-page__group-name">Разработка: команда 825</p>
              <p class="home-page__group-year">2025</p>
            </div>
          </div>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import {
  Cpu,
  DataAnalysis,
  FolderOpened,
  VideoPlay,
  Folder,
  Operation
} from '@element-plus/icons-vue'
</script>

<style scoped lang="scss">
.home-page {
  padding: 20px;
  min-height: calc(100vh - 40px);
  background: var(--el-bg-color-page);
  overflow-y: auto;

  &__card {
    max-width: 1200px;
    margin: 0 auto;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--el-border-color);
    background: var(--el-bg-color);
    max-height: calc(100vh - 80px);
    overflow-y: auto;
  }

  &__header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 5px 0;
    position: sticky;
    top: 0;
    background: var(--el-bg-color);
    z-index: 10;
    margin: -20px -20px 20px -20px;
    padding: 20px;
    border-bottom: 1px solid var(--el-border-color);
  }

  &__icon {
    color: var(--el-color-primary);
  }

  &__title {
    margin: 0;
    color: var(--el-text-color-primary);
    font-size: 1.6rem;
    font-weight: 600;
    line-height: 1.4;
  }

  &__content {
    padding: 10px 20px;
  }

  &__features {
    margin-bottom: 30px;
  }

  &__feature-col {
    margin-bottom: 20px;
  }

  &__feature-card {
    height: 100%;
    text-align: center;
    border: 1px solid var(--el-border-color);
    transition: transform 0.3s ease;
    border-radius: 10px;
    padding: 25px 20px;
    background: var(--el-bg-color);

    &:hover {
      transform: translateY(-3px);
      border-color: var(--el-color-primary);
      box-shadow: 0 4px 12px var(--el-box-shadow-light);
    }
  }

  &__feature-icon {
    margin-bottom: 20px;
  }

  &__feature-title {
    margin: 0 0 15px 0;
    color: var(--el-text-color-primary);
    font-size: 1.3rem;
    font-weight: 600;
  }

  &__feature-desc {
    color: var(--el-text-color-regular);
    line-height: 1.6;
    margin-bottom: 25px;
    font-size: 1rem;
    min-height: 60px;
  }

  &__process {
    margin-bottom: 30px;
    border: 1px solid var(--el-border-color);
    border-radius: 10px;
    background: var(--el-bg-color);
  }

  &__process-header {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--el-color-primary);

    .el-icon {
      font-size: 20px;
    }
  }

  &__process-steps {
    display: flex;
    flex-direction: column;
    gap: 20px;
    padding: 10px 0;
  }

  &__process-step {
    display: flex;
    align-items: flex-start;
    gap: 20px;
    padding: 15px;
    background: var(--el-fill-color-light);
    border-radius: 8px;
    border-left: 4px solid var(--el-color-primary);
  }

  &__step-number {
    width: 36px;
    height: 36px;
    background: var(--el-color-primary);
    color: var(--el-color-white);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 1.1rem;
    flex-shrink: 0;
  }

  &__step-content {
    flex: 1;
    
    h4 {
      margin: 0 0 8px 0;
      color: var(--el-text-color-primary);
      font-size: 1.1rem;
    }
    
    p {
      margin: 0;
      color: var(--el-text-color-regular);
      line-height: 1.5;
    }
  }

  &__tech-info {
    margin-bottom: 30px;
    border-radius: 8px;
    background: var(--el-bg-color);
    border: 1px solid var(--el-border-color);
    
    :deep(.el-alert__title) {
      color: var(--el-text-color-primary);
    }
    
    :deep(.el-alert__content) {
      width: 100%;
    }
    
    :deep(.el-alert__description) {
      color: var(--el-text-color-regular);
    }
  }

  &__tech-scroll {
    max-height: 200px;
    overflow-y: auto;
    padding-right: 10px;
    
    &::-webkit-scrollbar {
      width: 6px;
    }
    
    &::-webkit-scrollbar-track {
      background: var(--el-fill-color-light);
      border-radius: 3px;
    }
    
    &::-webkit-scrollbar-thumb {
      background: var(--el-border-color);
      border-radius: 3px;
      
      &:hover {
        background: var(--el-border-color-hover);
      }
    }
    
    p {
      margin: 8px 0;
      color: var(--el-text-color-regular);
      font-size: 0.95rem;
      padding: 4px 0;
    }
  }

  &__footer {
    margin-top: 40px;
    position: sticky;
    bottom: 0;
    background: var(--el-bg-color);
    padding-bottom: 10px;
    
    &-content {
      text-align: center;
    }
  }

  &__group-info {
    padding: 20px 0;
    
    .el-divider {
      margin: 20px 0;
      border-color: var(--el-border-color);
    }
  }

  &__group-name {
    font-size: 1.1rem;
    color: var(--el-color-primary);
    font-weight: 500;
    margin: 0 0 5px 0;
  }

  &__group-year {
    font-size: 0.95rem;
    color: var(--el-text-color-secondary);
    margin: 0;
  }
}

@media (max-width: 768px) {
  .home-page {
    padding: 10px;
    overflow-y: auto;

    &__card {
      max-height: calc(100vh - 20px);
      margin: 0;
    }

    &__title {
      font-size: 1.3rem;
    }

    &__header {
      margin: -10px -10px 15px -10px;
      padding: 15px;
    }

    &__content {
      padding: 10px;
    }

    &__process-step {
      flex-direction: column;
      align-items: flex-start;
      gap: 15px;
      border-left: none;
      border-top: 4px solid var(--el-color-primary);
    }

    &__tech-scroll {
      max-height: 150px;
    }
  }
}

/* Общие стили для скролла */
.home-page::-webkit-scrollbar,
.home-page__card::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

.home-page::-webkit-scrollbar-track,
.home-page__card::-webkit-scrollbar-track {
  background: var(--el-fill-color-light);
  border-radius: 4px;
}

.home-page::-webkit-scrollbar-thumb,
.home-page__card::-webkit-scrollbar-thumb {
  background: var(--el-border-color);
  border-radius: 4px;
  
  &:hover {
    background: var(--el-border-color-hover);
  }
}

/* Темная тема - дополнительные настройки */
@media (prefers-color-scheme: dark) {
  .home-page {
    &__feature-card:hover {
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    &__process-step {
      background: var(--el-fill-color-dark);
    }
  }
}
</style>