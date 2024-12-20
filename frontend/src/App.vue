<template>
  <div id="app">
    <LegalQuery />
  </div>
</template>

<script>
import LegalQuery from "./components/LegalQuery.vue";

export default {
  name: "App",
  components: {
    LegalQuery,
  },
};
</script>

<style>
#app {
  text-align: center;
}
</style>
