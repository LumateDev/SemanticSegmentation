<template>
  <header class="app-header">
    <div class="app-header__container">
      <!-- Логотип -->
      <router-link to="/" class="app-header__logo logo">
        <el-icon class="logo__icon">
          <Platform />
        </el-icon>
        <span class="logo__text">DGCNN</span>
      </router-link>

      <!-- Навигация -->
      <nav class="app-header__nav main-nav">
        <router-link
          v-for="item in navItems"
          :key="item.path"
          :to="item.path"
          class="main-nav__link"
          :class="{ 'main-nav__link--active': isActiveRoute(item.path) }"
        >
          {{ item.name }}
        </router-link>
      </nav>

      <!-- Переключатель темы -->
      <button class="app-header__theme-toggle theme-toggle" @click="toggleTheme">
        <el-icon class="theme-toggle__icon">
          <Moon v-if="isDark" />
          <Sunny v-else />
        </el-icon>
      </button>
    </div>
  </header>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useRoute } from 'vue-router';
import { Moon, Sunny, Platform } from '@element-plus/icons-vue';
import { useThemeStore } from '@/stores/themeStore';

const themeStore = useThemeStore();
const route = useRoute();

const isDark = computed(() => themeStore.isDark);

const navItems = [
  { path: '/models', name: 'Models' },
  { path: '/datasets', name: 'Datasets' },
  { path: '/test', name: 'Test' },
  { path: '/train', name: 'Train' },
  { path: '/predict', name: 'Predict' },
  { path: '/compare', name: 'Compare' },
];

const toggleTheme = () => {
  themeStore.toggleTheme();
};

const isActiveRoute = (path: string): boolean => {
  return route.path === path;
};
</script>

<style lang="scss" scoped>
.app-header {
  background: var(--el-bg-color);
  border-bottom: 1px solid var(--el-border-color-lighter);

  &__container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 24px;
    height: 60px;
    display: flex;
    align-items: center;
    gap: 32px;
  }

  &__nav {
    flex: 1;
  }
}

.logo {
  display: flex;
  align-items: center;
  gap: 8px;
  text-decoration: none;
  color: var(--el-text-color-primary);

  &__icon {
    font-size: 24px;
    color: var(--el-color-primary);
  }

  &__text {
    font-size: 18px;
    font-weight: 700;
  }
}

.main-nav {
  display: flex;
  align-items: center;
  gap: 8px;

  &__link {
    padding: 8px 16px;
    border-radius: 6px;
    text-decoration: none;
    color: var(--el-text-color-regular);
    font-size: 14px;
    font-weight: 500;

    &:hover {
      color: var(--el-color-primary);
      background: var(--el-color-primary-light-9);
    }

    &--active {
      color: var(--el-color-primary);
      background: var(--el-color-primary-light-9);
    }
  }
}

.theme-toggle {
  width: 36px;
  height: 36px;
  border-radius: 8px;
  border: 1px solid var(--el-border-color);
  background: var(--el-fill-color-light);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;

  &:hover {
    border-color: var(--el-color-primary);
  }

  &__icon {
    font-size: 18px;
    color: var(--el-text-color-regular);
  }
}
</style>
