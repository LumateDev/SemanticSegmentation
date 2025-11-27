<template>
  <el-menu mode="horizontal" class="app-header">
    <!-- Логотип / Home -->
    <router-link to="/" class="logo-link">
      <el-icon class="logo-icon"><Platform /></el-icon>
      <span class="logo-text">DGCNN</span>
    </router-link>

    <!-- Навигация -->
    <el-menu-item v-for="item in navItems" :key="item.path" :index="item.path">
      <router-link :to="item.path" class="nav-link">
        {{ item.name }}
      </router-link>
    </el-menu-item>

    <!-- Тогглер темы -->
    <div class="theme-toggle">
      <el-switch v-model="isDark" :active-icon="Moon" :inactive-icon="Sunny" />
    </div>
  </el-menu>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { Moon, Sunny, Platform } from '@element-plus/icons-vue'
import { useThemeStore } from '@/stores/themeStore'

const themeStore = useThemeStore()
const isDark = ref(themeStore.isDark)
watch(isDark, (val) => {
  themeStore.toggleTheme(val)
})

const navItems = [
  { path: '/models', name: 'Models' },
  { path: '/datasets', name: 'Datasets' },
  { path: '/test', name: 'Test' },
  { path: '/train', name: 'Train' },
  { path: '/predict', name: 'Predict' },
  { path: '/compare', name: 'Compare' },
]
</script>

<style scoped>
.app-header {
  border: none;
  padding: 0 1.5rem;
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.logo-link {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  text-decoration: none;
  color: var(--el-text-color-primary);
  font-weight: 600;
}

.logo-icon {
  font-size: 24px;
}

.nav-link {
  text-decoration: none;
  color: inherit;
  display: block;
  padding: 0.5rem 1rem;
}

.theme-toggle {
  margin-left: auto;
}
</style>
